//===-- lib/runtime/execute.cpp ---------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "flang/Runtime/execute.h"
#include "unit.h"
#include "flang-rt/runtime/descriptor.h"
#include "flang-rt/runtime/environment.h"
#include "flang-rt/runtime/stat.h"
#include "flang-rt/runtime/terminator.h"
#include "flang-rt/runtime/tools.h"
#include <cstdio>
#include <cstdlib>
#include <errno.h>
#include <future>
#include <limits>

#ifdef _WIN32
#include "flang/Common/windows-include.h"
#else
#include <signal.h>
#include <sys/wait.h>
#include <unistd.h>
#endif

namespace Fortran::runtime {

// cmdstat specified in 16.9.73
// −1 if the processor does not support command line execution,
// a processor-dependent positive value if an error condition occurs
// −2 if no error condition occurs but WAIT is present with the value false
// and the processor does not support asynchronous execution. Otherwise it is
// assigned the value 0
enum CMD_STAT {
  ASYNC_NO_SUPPORT_ERR = -2, // Linux setsid() returns -1
  NO_SUPPORT_ERR = -1, // system returns -1 with ENOENT
  CMD_EXECUTED = 0, // command executed with no error
  FORK_ERR = 1, // Linux fork() returns < 0
  EXECL_ERR = 2, // system returns -1 with other errno
  COMMAND_EXECUTION_ERR = 3, // Unexpected execution error
  COMMAND_CANNOT_EXECUTE_ERR = 4, // Linux exit code 126
  COMMAND_NOT_FOUND_ERR = 5, // Linux exit code 127
  INVALID_CL_ERR = 6, // cover all other non-zero exit code
  SIGNAL_ERR = 7
};

// Override CopyCharsToDescriptor in tools.h, pass string directly
void CopyCharsToDescriptor(const Descriptor &value, const char *rawValue) {
  CopyCharsToDescriptor(value, rawValue, std::strlen(rawValue));
}

void CheckAndCopyCharsToDescriptor(
    const Descriptor *value, const char *rawValue) {
  if (value) {
    CopyCharsToDescriptor(*value, rawValue);
  }
}

void CheckAndStoreIntToDescriptor(
    const Descriptor *intVal, std::int64_t value, Terminator &terminator) {
  if (intVal) {
    StoreIntToDescriptor(intVal, value, terminator);
  }
}

// If a condition occurs that would assign a nonzero value to CMDSTAT but
// the CMDSTAT variable is not present, error termination is initiated.
std::int64_t TerminationCheck(std::int64_t status, const Descriptor *cmdstat,
    const Descriptor *cmdmsg, Terminator &terminator) {
  // On both Windows and Linux, errno is set when system returns -1.
  if (status == -1) {
    // On Windows, ENOENT means the command interpreter can't be found.
    // On Linux, system calls execl with filepath "/bin/sh", ENOENT means the
    // file pathname does not exist.
    constexpr char msg[] = "Command line execution is not supported, system "
                           "returns -1 with errno ENOENT.";
    if (errno == ENOENT) {
      if (!cmdstat) {
        terminator.Crash(msg);
      } else {
        StoreIntToDescriptor(cmdstat, NO_SUPPORT_ERR, terminator);
        CheckAndCopyCharsToDescriptor(cmdmsg, msg);
        return status;
      }
    } else {
      char msg[256]{"Execution error with system status code: -1, errno: "};
      // Append the output of strerror*() to the end of msg. Note that upon
      // success, the output of strerror*() is always null-terminated.
      size_t appendIndex = std::strlen(msg);
#ifdef _WIN32
      if (strerror_s(msg + appendIndex, sizeof(msg) - appendIndex, errno) != 0)
#else
      if (strerror_r(errno, msg + appendIndex, sizeof(msg) - appendIndex) != 0)
#endif
        terminator.Crash("errno to char msg failed.");

      if (!cmdstat) {
        terminator.Crash(msg);
      } else {
        StoreIntToDescriptor(cmdstat, EXECL_ERR, terminator);
        CheckAndCopyCharsToDescriptor(cmdmsg, msg);
        return status;
      }
    }
  }

  // On WIN32 API std::system() returns exit status directly. On other OS'es,
  // special status codes are handled below.
  std::int64_t exitStatusVal{status};
#ifdef _WIN32
  if (status == 9009) {
    // cmd.exe returns status code 9009 for "command not found" error
    if (!cmdstat) {
      Crash(sourceFile, line, "Command not found.");
    } else {
      StoreIntToDescriptor(cmdstat, COMMAND_NOT_FOUND_ERR, sourceFile, line);
      CopyCharsToDescriptor(cmdmsg, "Command not found.", sourceFile, line);
    }
  }
#else

#if defined(WIFSIGNALED) && defined(WTERMSIG)
  if (WIFSIGNALED(status)) {
    if (!cmdstat) {
      terminator.Crash("Killed by signal: %d", WTERMSIG(status));
    } else {
      StoreIntToDescriptor(cmdstat, SIGNAL_ERR, terminator);
      CheckAndCopyCharsToDescriptor(cmdmsg, "Killed by signal");
      return WTERMSIG(status);
    }
  }
#endif

#if defined(WIFSTOPPED) && defined(WSTOPSIG)
  if (WIFSTOPPED(status)) {
    if (!cmdstat) {
      terminator.Crash("Stopped by signal: %d", WSTOPSIG(status));
    } else {
      StoreIntToDescriptor(cmdstat, SIGNAL_ERR, terminator);
      CheckAndCopyCharsToDescriptor(cmdmsg, "Stopped by signal");
      return WSTOPSIG(status);
    }
  }
#endif

#if defined(WIFEXITED) && defined(WEXITSTATUS)
  // WEXITSTATUS() returns valid value only if WIFEXITED(status) is true
  if (!WIFEXITED(status)) {
    if (!cmdstat) {
      terminator.Crash("Unexpected execution error: %d", status);
    } else {
      StoreIntToDescriptor(cmdstat, COMMAND_EXECUTION_ERR, terminator);
      CheckAndCopyCharsToDescriptor(cmdmsg, "Unexpected execution error");
      return status;
    }
  }
  exitStatusVal = WEXITSTATUS(status);
  // Status codes 126 and 127 are specific to Unix shell.
  if (exitStatusVal == 126) {
    if (!cmdstat) {
      terminator.Crash("Command cannot be executed with exit code: 126.");
    } else {
      StoreIntToDescriptor(cmdstat, COMMAND_CANNOT_EXECUTE_ERR, terminator);
      CheckAndCopyCharsToDescriptor(
          cmdmsg, "Command cannot be executed with exit code: 126.");
    }
  } else if (exitStatusVal == 127) {
    if (!cmdstat) {
      terminator.Crash("Command not found with exit code: 127.");
    } else {
      StoreIntToDescriptor(cmdstat, COMMAND_NOT_FOUND_ERR, terminator);
      CheckAndCopyCharsToDescriptor(
          cmdmsg, "Command not found with exit code: 127.");
    }
  }
#endif // WIFEXITED and WEXITSTATUS
#endif // Not _WIN32
  // At this point, any other status code is not known to be a "crashable
  // offense" and will be returned in EXITSTAT if provided.
  return exitStatusVal;
}

void RTNAME(ExecuteCommandLine)(const Descriptor &command, bool wait,
    const Descriptor *exitstat, const Descriptor *cmdstat,
    const Descriptor *cmdmsg, const char *sourceFile, int line) {
  Terminator terminator{sourceFile, line};
  char *newCmd{EnsureNullTerminated(
      command.OffsetElement(), command.ElementBytes(), terminator)};

  if (exitstat) {
    RUNTIME_CHECK(terminator, IsValidIntDescriptor(exitstat));
  }

  if (cmdstat) {
    RUNTIME_CHECK(terminator, IsValidIntDescriptor(cmdstat));
    // Assigned 0 as specifed in standard, if error then overwrite
    StoreIntToDescriptor(cmdstat, CMD_EXECUTED, terminator);
  }

  if (cmdmsg) {
    RUNTIME_CHECK(terminator, IsValidCharDescriptor(cmdmsg));
  }

  const char *cmd{newCmd};
#ifdef _WIN32
  // Construct a string that looks like
  //   "cmd.exe /v:on /c \"mycommand & exit /b !ERRORLEVEL!\""
  // Explanantion:
  //   /v:on - turns delayed environment variable expansion on, so
  //     variables written as !VAR! are expanded at execution time
  //     instead of at parse time. This is required for !ERRORLEVEL!
  //     to reflect the current error code at the moment exit runs.
  //   exit /b !ERRORLEVEL! - exits the current cmd instance (/b) and
  //     sets its process exit code to the current ERRORLEVEL value.
  //     Because delayed expansion is on, !ERRORLEVEL! is evaluated at
  //     execution time, so this cmd instance returns the same error
  //     code as mycommand.
  // This allows cmd.exe to either return the exit code of mycommand, or
  // to return its own exit code to the caller. The code 9009 is used
  // by cmd.exe to indicate "not found" condition.
  const char prefix[]{"cmd.exe /v:on /c \""};
  const char suffix[]{" & exit /b !ERRORLEVEL!\""};
  const size_t newCmdWinLen{
      (sizeof(prefix) - 1) + std::strlen(newCmd) + (sizeof(suffix) - 1) + 1};
  char *newCmdWin{
      static_cast<char *>(AllocateMemoryOrCrash(terminator, cmdBufLen))};
  std::snprintf(newCmdWin, newCmdWinLen, "%s%s%s", prefix, newCmd, suffix);
  cmd = newCmdWin;
#endif

  if (wait) {
    // either wait is not specified or wait is true: synchronous mode
    std::int64_t status{std::system(cmd)};
    std::int64_t exitStatusVal{
        TerminationCheck(status, cmdstat, cmdmsg, terminator)};
    // If sync, assigned processor-dependent exit status. Otherwise unchanged
    CheckAndStoreIntToDescriptor(exitstat, exitStatusVal, terminator);
  } else {
// Asynchronous mode
#ifdef _WIN32
    STARTUPINFOW si;
    PROCESS_INFORMATION pi;
    ZeroMemory(&si, sizeof(si));
    si.cb = sizeof(si);
    ZeroMemory(&pi, sizeof(pi));

    // Convert the char to wide char
    const size_t sizeNeeded{mbstowcs(NULL, newCmdWin, 0) + 1};
    wchar_t *wcmd{static_cast<wchar_t *>(
        AllocateMemoryOrCrash(terminator, sizeNeeded * sizeof(wchar_t)))};
    if (std::mbstowcs(wcmd, newCmdWin, sizeNeeded) == static_cast<size_t>(-1)) {
      terminator.Crash("Char to wide char failed for newCmd");
    }
    FreeMemory(newCmdWin);

    if (CreateProcessW(nullptr, wcmd, nullptr, nullptr, FALSE, 0, nullptr,
            nullptr, &si, &pi)) {
      // Close handles so it will be removed when terminated
      CloseHandle(pi.hProcess);
      CloseHandle(pi.hThread);
    } else {
      if (!cmdstat) {
        terminator.Crash(
            "CreateProcess failed with error code: %lu.", GetLastError());
      } else {
        StoreIntToDescriptor(cmdstat, ASYNC_NO_SUPPORT_ERR, terminator);
        CheckAndCopyCharsToDescriptor(cmdmsg, "CreateProcess failed.");
      }
    }
    FreeMemory(wcmd);
#else
    // Flush all the output streams before fork() in order to avoid parent's
    // buffered output to be replicated on the child. (Note: the issue of
    // duplicated output didn't happen for regular terminal output, but was
    // easy to reproduce when piping the output to a file.)
    io::IoErrorHandler handler{terminator};
    io::ExternalFileUnit::FlushAll(handler);
    std::fflush(nullptr); // Also flush stdio streams
    pid_t pid{fork()};
    if (pid < 0) {
      if (!cmdstat) {
        terminator.Crash("Fork failed with pid: %d.", pid);
      } else {
        StoreIntToDescriptor(cmdstat, FORK_ERR, terminator);
        CheckAndCopyCharsToDescriptor(cmdmsg, "Fork failed");
      }
    } else if (pid == 0) {
      // Create a new session, let init process take care of zombie child
      if (setsid() == -1) {
        if (!cmdstat) {
          terminator.Crash("setsid() failed with errno: %d, asynchronous "
                           "process initiation failed.",
              errno);
        } else {
          StoreIntToDescriptor(cmdstat, ASYNC_NO_SUPPORT_ERR, terminator);
          CheckAndCopyCharsToDescriptor(cmdmsg,
              "setsid() failed, asynchronous process initiation failed.");
        }
        exit(EXIT_FAILURE);
      }
      std::int64_t status{std::system(newCmd)};
      TerminationCheck(status, cmdstat, cmdmsg, terminator);
      exit(status);
    }
#endif
  }
  // Deallocate memory if EnsureNullTerminated dynamically allocated memory
  if (newCmd != command.OffsetElement()) {
    FreeMemory(newCmd);
  }
}

} // namespace Fortran::runtime
