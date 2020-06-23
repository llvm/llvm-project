//===-- strcmp_fuzz.cpp ---------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// Fuzzing test for llvm-libc strcmp implementation.
///
//===----------------------------------------------------------------------===//
#include "src/string/strcmp.h"
#include <stdint.h>

extern "C" int LLVMFuzzerTestTwoInputs(const uint8_t *data1, size_t size1,
                                       const uint8_t *data2, size_t size2) {
  // Verify each data source contains at least one character.
  if (!size1 || !size2)
    return 0;
  // Verify that the final character is the null terminator.
  if (data1[size1 - 1] != '\0' || data2[size2 - 1] != '\0')
    return 0;

  const char *s1 = reinterpret_cast<const char *>(data1);
  const char *s2 = reinterpret_cast<const char *>(data2);

  const size_t minimum_size = size1 < size2 ? size1 : size2;

  // Iterate through until either the minimum size is hit,
  // a character is the null terminator, or the first set
  // of differed bytes between s1 and s2 are found.
  // No bytes following a null byte should be compared.
  size_t i;
  for (i = 0; i < minimum_size; ++i) {
    if (!s1[i] || s1[i] != s2[i])
      break;
  }

  int expected_result = s1[i] - s2[i];
  int actual_result = __llvm_libc::strcmp(s1, s2);

  // The expected result should be the difference between the first non-equal
  // characters of s1 and s2. If all characters are equal, the expected result
  // should be '\0' - '\0' = 0.
  if (expected_result != actual_result)
    __builtin_trap();

  // Verify reversed operands. This should be the negated value of the previous
  // result, except of course if the previous result was zero.
  expected_result = s2[i] - s1[i];
  actual_result = __llvm_libc::strcmp(s2, s1);
  if (expected_result != actual_result)
    __builtin_trap();

  return 0;
}
