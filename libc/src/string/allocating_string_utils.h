//===-- Allocating string utils ---------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LIBC_SRC_STRING_ALLOCATING_STRING_UTILS_H
#define LIBC_SRC_STRING_ALLOCATING_STRING_UTILS_H

#include "src/__support/CPP/bitset.h"
#include "src/__support/common.h"
#include "src/string/memory_utils/bzero_implementations.h"
#include "src/string/memory_utils/memcpy_implementations.h"
#include "src/string/string_utils.h"
#include <stddef.h> // For size_t
#include <stdlib.h> // For malloc

namespace __llvm_libc {
namespace internal {

inline char *strdup(const char *src) {
  if (src == nullptr)
    return nullptr;
  size_t len = string_length(src) + 1;
  char *newstr = reinterpret_cast<char *>(::malloc(len));
  if (newstr == nullptr)
    return nullptr;
  inline_memcpy(newstr, src, len);
  return newstr;
}

} // namespace internal
} // namespace __llvm_libc

#endif
