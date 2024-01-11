//===-- bcmp_fuzz.cpp ---------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// Fuzzing test for llvm-libc bcmp implementation.
///
//===----------------------------------------------------------------------===//
#include "src/string/bcmp.h"
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

static int reference_bcmp(const void *pa, const void *pb, size_t count)
    __attribute__((no_builtin)) {
  const auto *a = reinterpret_cast<const unsigned char *>(pa);
  const auto *b = reinterpret_cast<const unsigned char *>(pb);
  for (size_t i = 0; i < count; ++i, ++a, ++b)
    if (*a != *b)
      return 1;
  return 0;
}

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size) {
  const auto normalize = [](int value) -> int {
    if (value == 0)
      return 0;
    return 1;
  };
  // We ignore the last byte is size is odd.
  const auto count = size / 2;
  const char *a = reinterpret_cast<const char *>(data);
  const char *b = reinterpret_cast<const char *>(data) + count;
  const int actual = LIBC_NAMESPACE::bcmp(a, b, count);
  const int reference = reference_bcmp(a, b, count);
  if (normalize(actual) == normalize(reference))
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
