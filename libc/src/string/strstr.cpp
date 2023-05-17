//===-- Implementation of strstr ------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/string/strstr.h"

#include "src/__support/common.h"
#include "src/string/memory_utils/strstr_implementations.h"

namespace __llvm_libc {

// TODO: This is a simple brute force implementation. This can be
// improved upon using well known string matching algorithms.
LLVM_LIBC_FUNCTION(char *, strstr, (const char *haystack, const char *needle)) {
  auto comp = [](char l, char r) -> int { return l - r; };
  return strstr_implementation(haystack, needle, comp);
}

} // namespace __llvm_libc
