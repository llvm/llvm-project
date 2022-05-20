//===-- Shared Converter Utilities for printf -------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_STDIO_PRINTF_CORE_CONVERTER_UTILS_H
#define LLVM_LIBC_SRC_STDIO_PRINTF_CORE_CONVERTER_UTILS_H

#include "src/__support/CPP/Limits.h"
#include "src/stdio/printf_core/core_structs.h"

#include <inttypes.h>
#include <stddef.h>

namespace __llvm_libc {
namespace printf_core {

inline uintmax_t apply_length_modifier(uintmax_t num, LengthModifier lm) {
  switch (lm) {
  case LengthModifier::none:
    return num & cpp::NumericLimits<unsigned int>::max();
  case LengthModifier::l:
    return num & cpp::NumericLimits<unsigned long>::max();
  case LengthModifier::ll:
  case LengthModifier::L:
    return num & cpp::NumericLimits<unsigned long long>::max();
  case LengthModifier::h:
    return num & cpp::NumericLimits<unsigned short>::max();
  case LengthModifier::hh:
    return num & cpp::NumericLimits<unsigned char>::max();
  case LengthModifier::z:
    return num & cpp::NumericLimits<size_t>::max();
  case LengthModifier::t:
    // We don't have unsigned ptrdiff so uintptr_t is used, since we need an
    // unsigned type and ptrdiff is usually the same size as a pointer.
    static_assert(sizeof(ptrdiff_t) == sizeof(uintptr_t));
    return num & cpp::NumericLimits<uintptr_t>::max();
  case LengthModifier::j:
    return num; // j is intmax, so no mask is necessary.
  }
}

#define RET_IF_RESULT_NEGATIVE(func)                                           \
  {                                                                            \
    int result = (func);                                                       \
    if (result < 0)                                                            \
      return result;                                                           \
  }

} // namespace printf_core
} // namespace __llvm_libc

#endif // LLVM_LIBC_SRC_STDIO_PRINTF_CORE_CONVERTER_UTILS_H
