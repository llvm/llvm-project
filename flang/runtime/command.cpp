//===-- runtime/command.cpp -----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "flang/Runtime/command.h"
#include "environment.h"
#include "stat.h"
#include "flang/Runtime/descriptor.h"
#include <cstdlib>
#include <limits>

namespace Fortran::runtime {
std::int32_t RTNAME(ArgumentCount)() {
  int argc{executionEnvironment.argc};
  if (argc > 1) {
    // C counts the command name as one of the arguments, but Fortran doesn't.
    return argc - 1;
  }
  return 0;
}

// Returns the length of the \p n'th argument. Assumes \p n is valid.
static std::int64_t ArgumentLength(std::int32_t n) {
  std::size_t length{std::strlen(executionEnvironment.argv[n])};
  if constexpr (sizeof(std::size_t) <= sizeof(std::int64_t)) {
    return static_cast<std::int64_t>(length);
  } else {
    std::size_t max{std::numeric_limits<std::int64_t>::max()};
    return length > max ? 0 // Just fail.
                        : static_cast<std::int64_t>(length);
  }
}

std::int64_t RTNAME(ArgumentLength)(std::int32_t n) {
  if (n < 0 || n >= executionEnvironment.argc) {
    return 0;
  }

  return ArgumentLength(n);
}

static bool IsValidCharDescriptor(const Descriptor *value) {
  return value && value->IsAllocated() &&
      value->type() == TypeCode(TypeCategory::Character, 1) &&
      value->rank() == 0;
}

static void FillWithSpaces(const Descriptor *value) {
  std::memset(value->OffsetElement(), ' ', value->ElementBytes());
}

std::int32_t RTNAME(ArgumentValue)(
    std::int32_t n, const Descriptor *value, const Descriptor *errmsg) {
  if (IsValidCharDescriptor(value)) {
    FillWithSpaces(value);
  }

  if (n < 0 || n >= executionEnvironment.argc) {
    return ToErrmsg(errmsg, StatInvalidArgumentNumber);
  }

  if (IsValidCharDescriptor(value)) {
    std::int64_t argLen{ArgumentLength(n)};
    if (argLen <= 0) {
      return ToErrmsg(errmsg, StatMissingArgument);
    }

    std::int64_t toCopy{
        std::min(argLen, static_cast<std::int64_t>(value->ElementBytes()))};
    std::memcpy(value->OffsetElement(), executionEnvironment.argv[n], toCopy);

    if (argLen > toCopy) {
      return ToErrmsg(errmsg, StatValueTooShort);
    }
  }

  return StatOk;
}

static std::size_t LengthWithoutTrailingSpaces(const Descriptor &d) {
  std::size_t s{d.ElementBytes() - 1};
  while (*d.OffsetElement(s) == ' ') {
    --s;
  }
  return s + 1;
}

std::int64_t RTNAME(EnvVariableLength)(const Descriptor &name, bool trim_name) {
  std::size_t nameLength{
      trim_name ? LengthWithoutTrailingSpaces(name) : name.ElementBytes()};
  if (nameLength == 0) {
    return 0;
  }

  const char *value{
      executionEnvironment.GetEnv(name.OffsetElement(), nameLength)};
  if (!value) {
    return 0;
  }
  return std::strlen(value);
}
} // namespace Fortran::runtime
