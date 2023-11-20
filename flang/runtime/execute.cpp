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
// #include <iostream>
// #include <string>
#include <limits>

namespace Fortran::runtime {

// Returns the length of the \p string. Assumes \p string is valid.
static std::int64_t StringLength(const char *string) {
  std::size_t length{std::strlen(string)};
  if constexpr (sizeof(std::size_t) < sizeof(std::int64_t)) {
    return static_cast<std::int64_t>(length);
  } else {
    std::size_t max{std::numeric_limits<std::int64_t>::max()};
    return length > max ? 0 // Just fail.
                        : static_cast<std::int64_t>(length);
  }
}

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

static std::int32_t CheckAndCopyToDescriptor(const Descriptor *value,
    const char *rawValue, const Descriptor *errmsg, std::size_t &offset) {
  bool haveValue{IsValidCharDescriptor(value)};

  std::int64_t len{StringLength(rawValue)};
  if (len <= 0) {
    if (haveValue) {
      FillWithSpaces(*value);
    }
    return ToErrmsg(errmsg, StatMissingArgument);
  }

  std::int32_t stat{StatOk};
  if (haveValue) {
    stat = CopyToDescriptor(*value, rawValue, len, errmsg, offset);
  }

  offset += len;
  return stat;
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

static bool FitsInDescriptor(
    const Descriptor *length, std::int64_t value, Terminator &terminator) {
  auto typeCode{length->type().GetCategoryAndKind()};
  int kind{typeCode->second};
  return Fortran::runtime::ApplyIntegerKind<FitsInIntegerKind, bool>(
      kind, terminator, value);
}

std::int32_t RTNAME(ExecuteCommandLine)(const Descriptor *command, bool wait,
    const Descriptor *exitstat, const Descriptor *cmdstat,
    const Descriptor *cmdmsg, const char *sourceFile, int line) {
  Terminator terminator{sourceFile, line};

  int exitstatVal;
  int cmdstatVal;

  if (wait) {
    // RUNTIME_CHECK(terminator, IsValidLogicalDescriptor(wait));
    // if (wait)
  } else {
    if (command) {
      RUNTIME_CHECK(terminator, IsValidCharDescriptor(command));
      exitstatVal = std::system(command->OffsetElement());
    }
  }

  if (exitstat) {
    RUNTIME_CHECK(terminator, IsValidIntDescriptor(exitstat));
    StoreIntToDescriptor(exitstat, exitstatVal, terminator);
  }

  if (cmdstat) {
    RUNTIME_CHECK(terminator, IsValidIntDescriptor(cmdstat));
    StoreIntToDescriptor(cmdstat, 0, terminator);
  }

  if (cmdmsg) {
    RUNTIME_CHECK(terminator, IsValidCharDescriptor(cmdmsg));
    std::array<char, 5> str;
    str.fill('f');
    CopyToDescriptor(*cmdmsg, str.data(), str.size(), nullptr);
  }

  // TODO

  return StatOk;
}

} // namespace Fortran::runtime
