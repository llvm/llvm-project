//===-- atof_fuzz.cpp -----------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// Fuzzing test for llvm-libc atof implementation.
///
//===----------------------------------------------------------------------===//
#include "src/stdlib/atof.h"
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>

#include "fuzzing/stdlib/StringParserOutputDiff.h"

// TODO: Remove this once glibc fixes hex subnormal rounding. See
// https://sourceware.org/bugzilla/show_bug.cgi?id=30220
#ifdef LLVM_LIBC_ATOF_DIF_FUZZ_SKIP_GLIBC_HEX_SUBNORMAL_ERR
#include <ctype.h>
constexpr double MIN_NORMAL = 0x1p-1022;

bool has_hex_prefix(const uint8_t *str) {
  size_t index = 0;

  // Skip over leading whitespace
  while (isspace(str[index])) {
    ++index;
  }
  // Skip over sign
  if (str[index] == '-' || str[index] == '+') {
    ++index;
  }
  return str[index] == '0' && (tolower(str[index + 1])) == 'x';
}

bool should_be_skipped(const uint8_t *str) {
  double init_result = ::atof(reinterpret_cast<const char *>(str));
  if (init_result < 0) {
    init_result = -init_result;
  }
  if (init_result < MIN_NORMAL && init_result != 0) {
    return has_hex_prefix(str);
  }
  return false;
}
#else
bool should_be_skipped(const uint8_t *) { return false; }
#endif // LLVM_LIBC_ATOF_DIF_FUZZ_SKIP_GLIBC_HEX_SUBNORMAL_ERR

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size) {
  uint8_t *container = new uint8_t[size + 1];
  if (!container)
    __builtin_trap();
  size_t i;

  for (i = 0; i < size; ++i)
    container[i] = data[i];
  container[size] = '\0'; // Add null terminator to container.

  if (!should_be_skipped(container)) {
    StringParserOutputDiff<double>(&__llvm_libc::atof, &::atof, container,
                                   size);
  }
  delete[] container;
  return 0;
}
