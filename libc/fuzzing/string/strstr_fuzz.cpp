//===-- strstr_fuzz.cpp ---------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// Fuzzing test for llvm-libc strstr implementation.
///
//===----------------------------------------------------------------------===//

#include "src/string/strlen.h"
#include "src/string/strstr.h"
#include <stddef.h>
#include <stdint.h>

// Simple loop to compare two strings up to a size n.
static int simple_memcmp(const char *left, const char *right, size_t n) {
  for (; n && *left == *right; ++left, ++right, --n)
    ;
  return n ? *left - *right : 0;
}

// The general structure is to take the value of the first byte, set size1 to
// that value, and add the null terminator. size2 will then contain the rest of
// the bytes in data.
// For example, with inputs (data={2, 6, 4, 8, 0}, size=5):
//         size1: data[0] = 2
//         data1: {2, 6} + '\0' = {2, 6, '\0'}
//         size2: size - size1 = 3
//         data2: {4, 8, '\0'}
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size) {
  // Verify the size is at least 1 and the data is null terminated.
  if (!size || data[size - 1] != '\0')
    return 0;
  const size_t size1 = (data[0] <= size ? data[0] : size);
  // The first size will always be at least 1 since
  // we need to append the null terminator. The second size
  // needs to be checked since it must also contain the null
  // terminator.
  if (size - size1 == 0)
    return 0;

  // Copy the data into a new container.
  uint8_t *container = new uint8_t[size1 + 1];
  if (!container)
    __builtin_trap();

  size_t i;
  for (i = 0; i < size1; ++i)
    container[i] = data[i];
  container[size1] = '\0'; // Add null terminator to container.

  const char *needle = reinterpret_cast<const char *>(container);
  const char *haystack = reinterpret_cast<const char *>(data + i);
  const char *result = LIBC_NAMESPACE::strstr(haystack, needle);

  // A null terminator may exist earlier in each, so this needs to be recorded.
  const size_t haystack_size = LIBC_NAMESPACE::strlen(haystack);
  const size_t needle_size = LIBC_NAMESPACE::strlen(needle);

  if (result) {
    // The needle is in the haystack.
    // 1. Verify that the result matches the needle.
    if (simple_memcmp(needle, result, needle_size) != 0)
      __builtin_trap();

    const char *haystack_ptr = haystack;
    // 2. Verify that the result is the first occurrence of the needle.
    for (; haystack_ptr != result; ++haystack_ptr) {
      if (simple_memcmp(needle, haystack_ptr, needle_size) == 0)
        __builtin_trap(); // There was an earlier occurrence of the needle.
    }
  } else {
    // No result was found. Verify that the needle doesn't exist within the
    // haystack.
    for (size_t i = 0; i + needle_size < haystack_size; ++i) {
      if (simple_memcmp(needle, haystack + i, needle_size) == 0)
        __builtin_trap(); // There was an earlier occurrence of the needle.
    }
  }
  delete[] container;
  return 0;
}
