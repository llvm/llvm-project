//===-- Allocating string utils ---------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_STRING_ALLOCATING_STRING_UTILS_H
#define LLVM_LIBC_SRC_STRING_ALLOCATING_STRING_UTILS_H

#include "src/__support/CPP/new.h"
#include "src/__support/CPP/optional.h"
#include "src/__support/macros/config.h"
#include "src/string/memory_utils/inline_memcpy.h"
#include "src/string/string_utils.h"

#include <stddef.h> // For size_t

namespace __llvm_libc {
namespace internal {

LIBC_INLINE cpp::optional<char *> strdup(const char *src) {
  if (src == nullptr)
    return cpp::nullopt;
  size_t len = string_length(src) + 1;
  AllocChecker ac;
  char *newstr = new (ac) char[len];
  if (!ac)
    return cpp::nullopt;
  inline_memcpy(newstr, src, len);
  return newstr;
}

} // namespace internal
} // namespace __llvm_libc

#endif // LLVM_LIBC_SRC_STRING_ALLOCATING_STRING_UTILS_H
