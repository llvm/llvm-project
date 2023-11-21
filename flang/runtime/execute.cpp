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

  int exitstatVal;
  int cmdstatVal;
  pid_t pid;
  std::array<char, 30> cmdstr;
  cmdstr.fill(' ');

  // cmdstat specified in 16.9.73
  // It is assigned the value −1 if the processor does not support command
  // line execution, a processor-dependent positive value if an error
  // condition occurs, or the value −2 if no error condition occurs but WAIT
  // is present with the value false and the processor does not support
  // asynchronous execution. Otherwise it is assigned the value 0
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

  if (wait) {
    // either wait is not specified or wait is true: synchronous mode
    exitstatVal = std::system(command->OffsetElement());
    cmdstatVal = CMD_EXECUTED;
  } else {
// Asynchronous mode, Windows doesn't support fork()
#ifdef _WIN32
    STARTUPINFO si;
    PROCESS_INFORMATION pi;

    ZeroMemory(&si, sizeof(si));
    si.cb = sizeof(si);
    ZeroMemory(&pi, sizeof(pi));

    if (CreateProcess(nullptr, const_cast<char *>(cmd), nullptr, nullptr, FALSE,
            0, nullptr, nullptr, &si, &pi)) {
      if (!GetExitCodeProcess(pi.hProcess, (DWORD)&exitstatVal)) {
        cmdstatVal = (uint32_t)GetLastError();
        std::strncpy(cmdstr.data(), "GetExitCodeProcess failed.", 26);
      } else {
        cmdstatVal = CMD_EXECUTED;
      }
    } else {
      cmdstatVal = (uint32_t)GetLastError();
      std::strncpy(cmdstr.data(), "CreateProcess failed.", 21);
    }

#else
    pid = fork();
    if (pid < 0) {
      std::strncpy(cmdstr.data(), "Fork failed", 11);
      cmdstatVal = FORK_ERR;
    } else if (pid == 0) {
      exitstatVal =
          execl("/bin/sh", "sh", "-c", command->OffsetElement(), (char *)NULL);
      cmdstatVal = CMD_EXECUTED;
      std::strncpy(cmdstr.data(), "Command executed.", 17);
    }
#endif
  }

  if (exitstat) {
    RUNTIME_CHECK(terminator, IsValidIntDescriptor(exitstat));
    StoreIntToDescriptor(exitstat, exitstatVal, terminator);
  }

  if (cmdstat) {
    RUNTIME_CHECK(terminator, IsValidIntDescriptor(cmdstat));
    StoreIntToDescriptor(cmdstat, cmdstatVal, terminator);
  }

  if (cmdmsg) {
    RUNTIME_CHECK(terminator, IsValidCharDescriptor(cmdmsg));
    FillWithSpaces(*cmdmsg);
    CopyToDescriptor(*cmdmsg, cmdstr.data(), cmdstr.size(), nullptr);
  }
}

} // namespace Fortran::runtime
