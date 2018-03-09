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
#include <fbl/unique_fd.h>
#include <fcntl.h>
#include <launchpad/launchpad.h>
#include <string>
#include <thread>
#include <zircon/errors.h>
#include <zircon/status.h>
#include <zircon/syscalls.h>
#include <zircon/syscalls/port.h>
#include <zircon/types.h>
#include <zx/object.h>
#include <zx/port.h>
#include <zx/process.h>
#include <zx/time.h>

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
  // Ctrl-C sends ETX in Zircon.
  while (getchar() != 0x03);
  Fuzzer::StaticInterruptCallback();
}

void CrashHandler(zx::port *Port) {
  std::unique_ptr<zx::port> ExceptionPort(Port);
  zx_port_packet_t Packet;
  ExceptionPort->wait(ZX_TIME_INFINITE, &Packet, 0);
  // Unbind as soon as possible so we don't receive exceptions from this thread.
  if (zx_task_bind_exception_port(ZX_HANDLE_INVALID, ZX_HANDLE_INVALID,
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
  zx::port *ExceptionPort = new zx::port();
  if ((rc = zx::port::create(0, ExceptionPort)) != ZX_OK) {
    Printf("libFuzzer: zx_port_create failed: %s\n", zx_status_get_string(rc));
    exit(1);
  }

  // Bind the port to receive exceptions from our process
  if ((rc = zx_task_bind_exception_port(zx_process_self(), ExceptionPort->get(),
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
  zx::nanosleep(zx::deadline_after(ZX_SEC(Seconds)));
}

unsigned long GetPid() {
  zx_status_t rc;
  zx_info_handle_basic_t Info;
  if ((rc = zx_object_get_info(zx_process_self(), ZX_INFO_HANDLE_BASIC, &Info,
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
  if ((rc = zx_object_get_info(zx_process_self(), ZX_INFO_TASK_STATS, &Info,
                               sizeof(Info), NULL, NULL)) != ZX_OK) {
    Printf("libFuzzer: unable to get info about self: %s\n",
           zx_status_get_string(rc));
    exit(1);
  }
  return (Info.mem_private_bytes + Info.mem_shared_bytes) >> 20;
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
  fbl::unique_fd OutputFile;
  if (Cmd.hasOutputFile()) {
    auto Filename = Cmd.getOutputFile();
    OutputFile.reset(open(Filename.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0));
    if (!OutputFile) {
      Printf("libFuzzer: failed to open %s: %s\n", Filename.c_str(),
             strerror(errno));
      return ZX_ERR_IO;
    }
    FdOut = OutputFile.get();
  }

  // Determine stderr
  int FdErr = STDERR_FILENO;
  if (Cmd.isOutAndErrCombined())
    FdErr = FdOut;

  // Clone the file descriptors into the new process
  if ((rc = launchpad_clone_fd(lp, STDIN_FILENO, STDIN_FILENO)) != ZX_OK ||
      (rc = launchpad_clone_fd(lp, FdOut, STDOUT_FILENO)) != ZX_OK ||
      (rc = launchpad_clone_fd(lp, FdErr, STDERR_FILENO)) != ZX_OK) {
    Printf("libFuzzer: failed to clone FDIO: %s\n", zx_status_get_string(rc));
    return rc;
  }

  // Start the process
  zx_handle_t ProcessHandle = ZX_HANDLE_INVALID;
  const char *ErrorMsg = nullptr;
  if ((rc = launchpad_go(lp, &ProcessHandle, &ErrorMsg)) != ZX_OK) {
    Printf("libFuzzer: failed to launch '%s': %s, %s\n", Argv[0], ErrorMsg,
           zx_status_get_string(rc));
    return rc;
  }
  zx::process Process(ProcessHandle);

  // Now join the process and return the exit status.
  if ((rc = Process.wait_one(ZX_PROCESS_TERMINATED, ZX_TIME_INFINITE,
                             nullptr)) != ZX_OK) {
    Printf("libFuzzer: failed to join '%s': %s\n", Argv[0],
           zx_status_get_string(rc));
    return rc;
  }

  zx_info_process_t Info;
  if ((rc = Process.get_info(ZX_INFO_PROCESS, &Info, sizeof(Info), nullptr,
                             nullptr)) != ZX_OK) {
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
