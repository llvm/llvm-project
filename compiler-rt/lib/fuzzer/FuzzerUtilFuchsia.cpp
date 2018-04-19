//===- FuzzerUtilFuchsia.cpp - Misc utils for Fuchsia. --------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
// Misc utils implementation using Fuchsia/Zircon APIs.
//===----------------------------------------------------------------------===//
#include "FuzzerDefs.h"

#if LIBFUZZER_FUCHSIA

#include "FuzzerInternal.h"
#include "FuzzerUtil.h"
#include <cerrno>
#include <cinttypes>
#include <cstdint>
#include <fcntl.h>
#include <launchpad/launchpad.h>
#include <string>
#include <thread>
#include <unistd.h>
#include <zircon/errors.h>
#include <zircon/process.h>
#include <zircon/status.h>
#include <zircon/syscalls.h>
#include <zircon/syscalls/port.h>
#include <zircon/types.h>

namespace fuzzer {

namespace {

// A magic value for the Zircon exception port, chosen to spell 'FUZZING'
// when interpreted as a byte sequence on little-endian platforms.
const uint64_t kFuzzingCrash = 0x474e495a5a5546;

void AlarmHandler(int Seconds) {
  while (true) {
    SleepSeconds(Seconds);
    Fuzzer::StaticAlarmCallback();
  }
}

void InterruptHandler() {
  fd_set readfds;
  // Ctrl-C sends ETX in Zircon.
  do {
    FD_ZERO(&readfds);
    FD_SET(STDIN_FILENO, &readfds);
    select(STDIN_FILENO + 1, &readfds, nullptr, nullptr, nullptr);
  } while(!FD_ISSET(STDIN_FILENO, &readfds) || getchar() != 0x03);
  Fuzzer::StaticInterruptCallback();
}

void CrashHandler(zx_handle_t *Port) {
  std::unique_ptr<zx_handle_t> ExceptionPort(Port);
  zx_port_packet_t Packet;
  _zx_port_wait(*ExceptionPort, ZX_TIME_INFINITE, &Packet, 1);
  // Unbind as soon as possible so we don't receive exceptions from this thread.
  if (_zx_task_bind_exception_port(ZX_HANDLE_INVALID, ZX_HANDLE_INVALID,
                                   kFuzzingCrash, 0) != ZX_OK) {
    // Shouldn't happen; if it does the safest option is to just exit.
    Printf("libFuzzer: unable to unbind exception port; aborting!\n");
    exit(1);
  }
  if (Packet.key != kFuzzingCrash) {
    Printf("libFuzzer: invalid crash key: %" PRIx64 "; aborting!\n",
           Packet.key);
    exit(1);
  }
  // CrashCallback should not return from this call
  Fuzzer::StaticCrashSignalCallback();
}

} // namespace

// Platform specific functions.
void SetSignalHandler(const FuzzingOptions &Options) {
  zx_status_t rc;

  // Set up alarm handler if needed.
  if (Options.UnitTimeoutSec > 0) {
    std::thread T(AlarmHandler, Options.UnitTimeoutSec / 2 + 1);
    T.detach();
  }

  // Set up interrupt handler if needed.
  if (Options.HandleInt || Options.HandleTerm) {
    std::thread T(InterruptHandler);
    T.detach();
  }

  // Early exit if no crash handler needed.
  if (!Options.HandleSegv && !Options.HandleBus && !Options.HandleIll &&
      !Options.HandleFpe && !Options.HandleAbrt)
    return;

  // Create an exception port
  zx_handle_t *ExceptionPort = new zx_handle_t;
  if ((rc = _zx_port_create(0, ExceptionPort)) != ZX_OK) {
    Printf("libFuzzer: zx_port_create failed: %s\n", _zx_status_get_string(rc));
    exit(1);
  }

  // Bind the port to receive exceptions from our process
  if ((rc = _zx_task_bind_exception_port(_zx_process_self(), *ExceptionPort,
                                         kFuzzingCrash, 0)) != ZX_OK) {
    Printf("libFuzzer: unable to bind exception port: %s\n",
           zx_status_get_string(rc));
    exit(1);
  }

  // Set up the crash handler.
  std::thread T(CrashHandler, ExceptionPort);
  T.detach();
}

void SleepSeconds(int Seconds) {
  _zx_nanosleep(_zx_deadline_after(ZX_SEC(Seconds)));
}

unsigned long GetPid() {
  zx_status_t rc;
  zx_info_handle_basic_t Info;
  if ((rc = zx_object_get_info(_zx_process_self(), ZX_INFO_HANDLE_BASIC, &Info,
                               sizeof(Info), NULL, NULL)) != ZX_OK) {
    Printf("libFuzzer: unable to get info about self: %s\n",
           zx_status_get_string(rc));
    exit(1);
  }
  return Info.koid;
}

size_t GetPeakRSSMb() {
  zx_status_t rc;
  zx_info_task_stats_t Info;
  if ((rc = _zx_object_get_info(_zx_process_self(), ZX_INFO_TASK_STATS, &Info,
                               sizeof(Info), NULL, NULL)) != ZX_OK) {
    Printf("libFuzzer: unable to get info about self: %s\n",
           _zx_status_get_string(rc));
    exit(1);
  }
  return (Info.mem_private_bytes + Info.mem_shared_bytes) >> 20;
}

template <typename Fn>
class RunOnDestruction {
 public:
  explicit RunOnDestruction(Fn fn) : fn_(fn) {}
  ~RunOnDestruction() { fn_(); }

 private:
  Fn fn_;
};

template <typename Fn>
RunOnDestruction<Fn> at_scope_exit(Fn fn) {
  return RunOnDestruction<Fn>(fn);
}

int ExecuteCommand(const Command &Cmd) {
  zx_status_t rc;

  // Convert arguments to C array
  auto Args = Cmd.getArguments();
  size_t Argc = Args.size();
  assert(Argc != 0);
  std::unique_ptr<const char *[]> Argv(new const char *[Argc]);
  for (size_t i = 0; i < Argc; ++i)
    Argv[i] = Args[i].c_str();

  // Create the basic launchpad.  Clone everything except stdio.
  launchpad_t *lp;
  launchpad_create(ZX_HANDLE_INVALID, Argv[0], &lp);
  launchpad_load_from_file(lp, Argv[0]);
  launchpad_set_args(lp, Argc, Argv.get());
  launchpad_clone(lp, LP_CLONE_ALL & (~LP_CLONE_FDIO_STDIO));

  // Determine stdout
  int FdOut = STDOUT_FILENO;

  if (Cmd.hasOutputFile()) {
    auto Filename = Cmd.getOutputFile();
    FdOut = open(Filename.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0);
    if (FdOut == -1) {
      Printf("libFuzzer: failed to open %s: %s\n", Filename.c_str(),
             strerror(errno));
      return ZX_ERR_IO;
    }
  }
  auto CloseFdOut = at_scope_exit([&]() { close(FdOut); } );

  // Determine stderr
  int FdErr = STDERR_FILENO;
  if (Cmd.isOutAndErrCombined())
    FdErr = FdOut;

  // Clone the file descriptors into the new process
  if ((rc = launchpad_clone_fd(lp, STDIN_FILENO, STDIN_FILENO)) != ZX_OK ||
      (rc = launchpad_clone_fd(lp, FdOut, STDOUT_FILENO)) != ZX_OK ||
      (rc = launchpad_clone_fd(lp, FdErr, STDERR_FILENO)) != ZX_OK) {
    Printf("libFuzzer: failed to clone FDIO: %s\n", _zx_status_get_string(rc));
    return rc;
  }

  // Start the process
  zx_handle_t ProcessHandle = ZX_HANDLE_INVALID;
  const char *ErrorMsg = nullptr;
  if ((rc = launchpad_go(lp, &ProcessHandle, &ErrorMsg)) != ZX_OK) {
    Printf("libFuzzer: failed to launch '%s': %s, %s\n", Argv[0], ErrorMsg,
           _zx_status_get_string(rc));
    return rc;
  }
  auto CloseHandle = at_scope_exit([&]() { _zx_handle_close(ProcessHandle); });

  // Now join the process and return the exit status.
  if ((rc = _zx_object_wait_one(ProcessHandle, ZX_PROCESS_TERMINATED,
                                ZX_TIME_INFINITE, nullptr)) != ZX_OK) {
    Printf("libFuzzer: failed to join '%s': %s\n", Argv[0],
           _zx_status_get_string(rc));
    return rc;
  }

  zx_info_process_t Info;
  if ((rc = _zx_object_get_info(ProcessHandle, ZX_INFO_PROCESS, &Info,
                                sizeof(Info), nullptr, nullptr)) != ZX_OK) {
    Printf("libFuzzer: unable to get return code from '%s': %s\n", Argv[0],
           zx_status_get_string(rc));
    return rc;
  }

  return Info.return_code;
}

const void *SearchMemory(const void *Data, size_t DataLen, const void *Patt,
                         size_t PattLen) {
  return memmem(Data, DataLen, Patt, PattLen);
}

} // namespace fuzzer

#endif // LIBFUZZER_FUCHSIA
