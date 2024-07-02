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
  ASYNC_NO_SUPPORT_ERR = -2, // system returns -1 with ENOENT
  NO_SUPPORT_ERR = -1, // Linux setsid() returns -1
  CMD_EXECUTED = 0, // command executed with no error
  FORK_ERR = 1, // Linux fork() returns < 0
  EXECL_ERR = 2, // system returns -1 with other errno
  COMMAND_EXECUTION_ERR = 3, // exit code 1
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
    if (errno == ENOENT) {
      if (!cmdstat) {
        terminator.Crash("Command line execution is not supported, system "
                         "returns -1 with errno ENOENT.");
      } else {
        StoreIntToDescriptor(cmdstat, NO_SUPPORT_ERR, terminator);
        CheckAndCopyCharsToDescriptor(cmdmsg,
            "Command line execution is not supported, system returns -1 with "
            "errno ENOENT.");
      }
    } else {
      char err_buffer[30];
      char msg[]{"Execution error with system status code: -1, errno: "};
#ifdef _WIN32
      if (strerror_s(err_buffer, sizeof(err_buffer), errno) != 0)
#else
      if (strerror_r(errno, err_buffer, sizeof(err_buffer)) != 0)
#endif
        terminator.Crash("errno to char msg failed.");
      char *newMsg{static_cast<char *>(AllocateMemoryOrCrash(
          terminator, std::strlen(msg) + std::strlen(err_buffer) + 1))};
      std::strcat(newMsg, err_buffer);

      if (!cmdstat) {
        terminator.Crash(newMsg);
      } else {
        StoreIntToDescriptor(cmdstat, EXECL_ERR, terminator);
        CheckAndCopyCharsToDescriptor(cmdmsg, newMsg);
      }
      FreeMemory(newMsg);
    }
  }

#ifdef _WIN32
  // On WIN32 API std::system returns exit status directly
  std::int64_t exitStatusVal{status};
  if (exitStatusVal != 0) {
    if (!cmdstat) {
      terminator.Crash(
          "Invalid command quit with exit status code: %d", exitStatusVal);
    } else {
      StoreIntToDescriptor(cmdstat, INVALID_CL_ERR, terminator);
      CheckAndCopyCharsToDescriptor(cmdmsg, "Invalid command line");
    }
  }
#else
  std::int64_t exitStatusVal{WEXITSTATUS(status)};
  if (exitStatusVal == 1) {
    if (!cmdstat) {
      terminator.Crash("Command line execution failed with exit code: 1.");
    } else {
      StoreIntToDescriptor(cmdstat, COMMAND_EXECUTION_ERR, terminator);
      CheckAndCopyCharsToDescriptor(
          cmdmsg, "Command line execution failed with exit code: 1.");
    }
  } else if (exitStatusVal == 126) {
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
    // capture all other nonzero exit code
  } else if (exitStatusVal != 0) {
    if (!cmdstat) {
      terminator.Crash(
          "Invalid command quit with exit status code: %d", exitStatusVal);
    } else {
      StoreIntToDescriptor(cmdstat, INVALID_CL_ERR, terminator);
      CheckAndCopyCharsToDescriptor(cmdmsg, "Invalid command line");
    }
  }
#endif

#if defined(WIFSIGNALED) && defined(WTERMSIG)
  if (WIFSIGNALED(status)) {
    if (!cmdstat) {
      terminator.Crash("Killed by signal: %d", WTERMSIG(status));
    } else {
      StoreIntToDescriptor(cmdstat, SIGNAL_ERR, terminator);
      CheckAndCopyCharsToDescriptor(cmdmsg, "Killed by signal");
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
    std::int64_t status{std::system(newCmd)};
    std::int64_t exitStatusVal{
        TerminationCheck(status, cmdstat, cmdmsg, terminator)};
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
        StoreIntToDescriptor(cmdstat, ASYNC_NO_SUPPORT_ERR, terminator);
        CheckAndCopyCharsToDescriptor(cmdmsg, "CreateProcess failed.");
      }
    }
    FreeMemory(wcmd);
#else
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
