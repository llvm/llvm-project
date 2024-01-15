//===-- runtime/execute.cpp -----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "flang/Runtime/execute.h"
#include "environment.h"
#include "stat.h"
#include "terminator.h"
#include "tools.h"
#include "flang/Runtime/descriptor.h"
#include <cstdlib>
#include <future>
#include <limits>
#ifdef _WIN32
#define LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
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
  ASYNC_NO_SUPPORT_ERR = -2,
  NO_SUPPORT_ERR = -1,
  CMD_EXECUTED = 0,
  FORK_ERR = 1,
  EXECL_ERR = 2,
  INVALID_CL_ERR = 3,
  SIGNAL_ERR = 4
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
int TerminationCheck(int status, const Descriptor *cmdstat,
    const Descriptor *cmdmsg, Terminator &terminator) {
  if (status == -1) {
    if (!cmdstat) {
      terminator.Crash("Execution error with system status code: %d", status);
    } else {
      StoreIntToDescriptor(cmdstat, EXECL_ERR, terminator);
      CheckAndCopyCharsToDescriptor(cmdmsg, "Execution error");
    }
  }
#ifdef _WIN32
  // On WIN32 API std::system returns exit status directly
  int exitStatusVal{status};
  if (exitStatusVal == 1) {
#else
  int exitStatusVal{WEXITSTATUS(status)};
  if (exitStatusVal == 127 || exitStatusVal == 126) {
#endif
    if (!cmdstat) {
      terminator.Crash(
          "Invalid command quit with exit status code: %d", exitStatusVal);
    } else {
      StoreIntToDescriptor(cmdstat, INVALID_CL_ERR, terminator);
      CheckAndCopyCharsToDescriptor(cmdmsg, "Invalid command line");
    }
  }
#if defined(WIFSIGNALED) && defined(WTERMSIG)
  if (WIFSIGNALED(status)) {
    if (!cmdstat) {
      terminator.Crash("killed by signal: %d", WTERMSIG(status));
    } else {
      StoreIntToDescriptor(cmdstat, SIGNAL_ERR, terminator);
      CheckAndCopyCharsToDescriptor(cmdmsg, "killed by signal");
    }
  }
#endif
#if defined(WIFSTOPPED) && defined(WSTOPSIG)
  if (WIFSTOPPED(status)) {
    if (!cmdstat) {
      terminator.Crash("stopped by signal: %d", WSTOPSIG(status));
    } else {
      StoreIntToDescriptor(cmdstat, SIGNAL_ERR, terminator);
      CheckAndCopyCharsToDescriptor(cmdmsg, "stopped by signal");
    }
  }
#endif
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

  if (wait) {
    // either wait is not specified or wait is true: synchronous mode
    int status{std::system(newCmd)};
    int exitStatusVal{TerminationCheck(status, cmdstat, cmdmsg, terminator)};
    // If sync, assigned processor-dependent exit status. Otherwise unchanged
    CheckAndStoreIntToDescriptor(exitstat, exitStatusVal, terminator);
  } else {
// Asynchronous mode
#ifdef _WIN32
    STARTUPINFO si;
    PROCESS_INFORMATION pi;
    ZeroMemory(&si, sizeof(si));
    si.cb = sizeof(si);
    ZeroMemory(&pi, sizeof(pi));

    // add "cmd.exe /c " to the beginning of command
    const char *prefix{"cmd.exe /c "};
    char *newCmdWin{static_cast<char *>(AllocateMemoryOrCrash(
        terminator, std::strlen(prefix) + std::strlen(newCmd) + 1))};
    std::strcpy(newCmdWin, prefix);
    std::strcat(newCmdWin, newCmd);

    // Convert the char to wide char
    const size_t sizeNeeded{mbstowcs(NULL, newCmdWin, 0) + 1};
    wchar_t *wcmd{static_cast<wchar_t *>(
        AllocateMemoryOrCrash(terminator, sizeNeeded * sizeof(wchar_t)))};
    if (std::mbstowcs(wcmd, newCmdWin, sizeNeeded) == static_cast<size_t>(-1)) {
      terminator.Crash("Char to wide char failed for newCmd");
    }
    FreeMemory(newCmdWin);

    if (CreateProcess(nullptr, wcmd, nullptr, nullptr, FALSE, 0, nullptr,
            nullptr, &si, &pi)) {
      // Close handles so it will be removed when terminated
      CloseHandle(pi.hProcess);
      CloseHandle(pi.hThread);
    } else {
      if (!cmdstat) {
        terminator.Crash(
            "CreateProcess failed with error code: %lu.", GetLastError());
      } else {
        StoreIntToDescriptor(cmdstat, (uint32_t)GetLastError(), terminator);
        CheckAndCopyCharsToDescriptor(cmdmsg, "CreateProcess failed.");
      }
    }
    FreeMemory(wcmd);
#else
    // terminated children do not become zombies
    signal(SIGCHLD, SIG_IGN);
    pid_t pid{fork()};
    if (pid < 0) {
      if (!cmdstat) {
        terminator.Crash("Fork failed with pid: %d.", pid);
      } else {
        StoreIntToDescriptor(cmdstat, FORK_ERR, terminator);
        CheckAndCopyCharsToDescriptor(cmdmsg, "Fork failed");
      }
    } else if (pid == 0) {
      int status{std::system(newCmd)};
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
