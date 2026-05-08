//===-- strnlen_s_differential_fuzz.cpp -----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// Differential fuzz test for llvm-libc strnlen_s implementation.
///
//===----------------------------------------------------------------------===//

#define __STDC_WANT_LIB_EXT1__ 1

#include "src/string/strnlen_s.h"
#include <stdint.h>
#include <string.h>

extern "C" size_t LLVMFuzzerMutate(uint8_t *data, size_t size, size_t max_size);
extern "C" size_t LLVMFuzzerCustomMutator(uint8_t *data, size_t size,
                                          size_t max_size,
                                          unsigned int /*seed*/) {
  // The buffer is constructed as follows:
  // data = max_len (size_t) + null-terminated string
  if (max_size < sizeof(size_t) + 1)
    return size;

  do {
    size = LLVMFuzzerMutate(data, size, max_size);
  } while (size < sizeof(size_t) + 1);

  data[size - 1] = '\0';
  return size;
}

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size) {
  if (size < sizeof(size_t) + 1)
    return 0;

  size_t max_len;
  ::memcpy(&max_len, data, sizeof(size_t));
  data += sizeof(size_t);

  // If Annex K is not available in the system's C library, we compare against
  // strnlen instead. We can assume this is valid because in the case where the
  // input string is not null, the two functions must have identical semantics.
#ifdef __STDC_LIB_EXT1__
  size_t ref = ::strnlen_s(reinterpret_cast<const char *>(data), max_len);
#else
  size_t ref = ::strnlen(reinterpret_cast<const char *>(data), max_len);
#endif
  size_t impl =
      LIBC_NAMESPACE::strnlen_s(reinterpret_cast<const char *>(data), max_len);

  if (ref != impl)
    __builtin_trap();

  return 0;
}
