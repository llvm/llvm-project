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
#include <unistd.h>
#endif

namespace Fortran::runtime {

static bool IsValidCharDescriptor(const Descriptor *value) {
  return value && value->IsAllocated() &&
      value->type() == TypeCode(TypeCategory::Character, 1) &&
      value->rank() == 0;
}

static bool IsValidIntDescriptor(const Descriptor *length) {
  auto typeCode{length->type().GetCategoryAndKind()};
  // Check that our descriptor is allocated and is a scalar integer with
  // kind != 1 (i.e. with a large enough decimal exponent range).
  return length->IsAllocated() && length->rank() == 0 &&
      length->type().IsInteger() && typeCode && typeCode->second != 1;
}

static void FillWithSpaces(const Descriptor &value, std::size_t offset = 0) {
  if (offset < value.ElementBytes()) {
    std::memset(
        value.OffsetElement(offset), ' ', value.ElementBytes() - offset);
  }
}

static std::int32_t CopyToDescriptor(const Descriptor &value,
    const char *rawValue, std::int64_t rawValueLength, const Descriptor *errmsg,
    std::size_t offset = 0) {

  std::int64_t toCopy{std::min(rawValueLength,
      static_cast<std::int64_t>(value.ElementBytes() - offset))};
  if (toCopy < 0) {
    return ToErrmsg(errmsg, StatValueTooShort);
  }

  std::memcpy(value.OffsetElement(offset), rawValue, toCopy);

  if (rawValueLength > toCopy) {
    return ToErrmsg(errmsg, StatValueTooShort);
  }

  return StatOk;
}

static void StoreIntToDescriptor(
    const Descriptor *intVal, std::int64_t value, Terminator &terminator) {
  auto typeCode{intVal->type().GetCategoryAndKind()};
  int kind{typeCode->second};
  Fortran::runtime::ApplyIntegerKind<Fortran::runtime::StoreIntegerAt, void>(
      kind, terminator, *intVal, /* atIndex = */ 0, value);
}

template <int KIND> struct FitsInIntegerKind {
  bool operator()([[maybe_unused]] std::int64_t value) {
    if constexpr (KIND >= 8) {
      return true;
    } else {
      return value <= std::numeric_limits<Fortran::runtime::CppTypeFor<
                          Fortran::common::TypeCategory::Integer, KIND>>::max();
    }
  }
};

void RTNAME(ExecuteCommandLine)(const Descriptor *command, bool wait,
    const Descriptor *exitstat, const Descriptor *cmdstat,
    const Descriptor *cmdmsg, const char *sourceFile, int line) {
  Terminator terminator{sourceFile, line};

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
    SIGNAL_ERR = 3
  };

  if (command) {
    RUNTIME_CHECK(terminator, IsValidCharDescriptor(command));
  }
  if (exitstat) {
    RUNTIME_CHECK(terminator, IsValidIntDescriptor(exitstat));
    // If sync, assigned processor-dependent exit status. Otherwise unchanged
  }

  if (cmdstat) {
    RUNTIME_CHECK(terminator, IsValidIntDescriptor(cmdstat));
    // If a condition occurs that would assign a nonzero value to CMDSTAT but
    // the CMDSTAT variable is not present, error termination is initiated.
    // Assigned 0 as specifed in standard, if error then overwrite
    StoreIntToDescriptor(cmdstat, CMD_EXECUTED, terminator);
  }

  if (cmdmsg) {
    RUNTIME_CHECK(terminator, IsValidCharDescriptor(cmdmsg));
  }

  if (wait) {
    // either wait is not specified or wait is true: synchronous mode
    int exitstatVal = std::system(command->OffsetElement());
    StoreIntToDescriptor(exitstat, exitstatVal, terminator);
  } else {
// Asynchronous mode
#ifdef _WIN32
    STARTUPINFO si;
    PROCESS_INFORMATION pi;
    ZeroMemory(&si, sizeof(si));
    si.cb = sizeof(si);
    ZeroMemory(&pi, sizeof(pi));

    // append "cmd.exe /c " to the begining of command
    const char *cmd = command->OffsetElement();
    const char *prefix = "cmd.exe /c ";
    char *newCmd = (char *)malloc(strlen(prefix) + strlen(cmd) + 1);
    if (newCmd != NULL) {
      std::strcpy(newCmd, prefix);
      std::strcat(newCmd, cmd);
    } else {
      terminator.Crash("Memory allocation failed for newCmd");
    }

    // Convert the narrow string to a wide string
    int size_needed = MultiByteToWideChar(CP_UTF8, 0, newCmd, -1, NULL, 0);
    wchar_t *wcmd = new wchar_t[size_needed];
    if (MultiByteToWideChar(CP_UTF8, 0, newCmd, -1, wcmd, size_needed) != 0) {
      terminator.Crash(
          "Char to wider char conversion failed with error code: %lu.",
          GetLastError());
    }
    free(newCmd);

    if (!CreateProcess(nullptr, wcmd, nullptr, nullptr, FALSE, 0, nullptr,
            nullptr, &si, &pi)) {
      if (!cmdstat) {
        terminator.Crash(
            "CreateProcess failed with error code: %lu.", GetLastError());
      } else {
        StoreIntToDescriptor(cmdstat, (uint32_t)GetLastError(), terminator);
        CopyToDescriptor(*cmdmsg, "CreateProcess failed.", 21, nullptr);
      }
    }
    delete[] wcmd;
#else
    pid_t pid = fork();
    if (pid < 0) {
      if (!cmdstat) {
        terminator.Crash("Fork failed with error code: %d.", FORK_ERR);
      } else {
        StoreIntToDescriptor(cmdstat, FORK_ERR, terminator);
        CopyToDescriptor(*cmdmsg, "Fork failed", 11, nullptr);
      }
    } else if (pid == 0) {
      execl("/bin/sh", "sh", "-c", command->OffsetElement(), (char *)NULL);
    }
#endif
  }
}

} // namespace Fortran::runtime
