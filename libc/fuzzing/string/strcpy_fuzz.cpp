//===-- strcpy_fuzz.cpp ---------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// Fuzzing test for llvm-libc strcpy implementation.
///
//===----------------------------------------------------------------------===//
#include "src/string/strcpy.h"
#include <stdint.h>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size) {
  // Validate input
  if (!size) return 0;
  if (data[size - 1] != '\0') return 0;
  const char *src = (const char *)data;

  char *dest = new char[size];
  if (!dest) __builtin_trap();

  LIBC_NAMESPACE::strcpy(dest, src);

  size_t i;
  for (i = 0; src[i] != '\0'; i++) {
    // Ensure correctness of strcpy
    if (dest[i] != src[i]) __builtin_trap();
  }
  // Ensure strcpy null terminates dest
  if (dest[i] != src[i]) __builtin_trap();

  delete[] dest;

  return 0;
}

