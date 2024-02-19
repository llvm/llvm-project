//===-- memcmp_fuzz.cpp ---------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// Fuzzing test for llvm-libc memcmp implementation.
///
//===----------------------------------------------------------------------===//
#include "src/string/memcmp.h"
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

static int reference_memcmp(const void *pa, const void *pb, size_t count)
    __attribute__((no_builtin)) {
  const auto *a = reinterpret_cast<const unsigned char *>(pa);
  const auto *b = reinterpret_cast<const unsigned char *>(pb);
  for (size_t i = 0; i < count; ++i, ++a, ++b) {
    if (*a < *b)
      return -1;
    else if (*a > *b)
      return 1;
  }
  return 0;
}

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size) {
  const auto sign = [](int value) -> int {
    if (value < 0)
      return -1;
    if (value > 0)
      return 1;
    return 0;
  };
  // We ignore the last byte is size is odd.
  const auto count = size / 2;
  const char *a = reinterpret_cast<const char *>(data);
  const char *b = reinterpret_cast<const char *>(data) + count;
  const int actual = LIBC_NAMESPACE::memcmp(a, b, count);
  const int reference = reference_memcmp(a, b, count);
  if (sign(actual) == sign(reference))
    return 0;
  const auto print = [](const char *msg, const char *buffer, size_t size) {
    printf("%s\"", msg);
    for (size_t i = 0; i < size; ++i)
      printf("\\x%02x", (uint8_t)buffer[i]);
    printf("\"\n");
  };
  printf("count    : %zu\n", count);
  print("a        : ", a, count);
  print("b        : ", b, count);
  printf("expected : %d\n", reference);
  printf("actual   : %d\n", actual);
  __builtin_trap();
}
