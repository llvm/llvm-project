//===-- strtofloat_fuzz.cpp -----------------------------------------------===//
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
#include "src/stdlib/strtod.h"
#include "src/stdlib/strtof.h"
#include "src/stdlib/strtold.h"
#include <math.h>
#include <stddef.h>
#include <stdint.h>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size) {
  uint8_t *container = new uint8_t[size + 1];
  if (!container)
    __builtin_trap();
  size_t i;

  for (i = 0; i < size; ++i)
    container[i] = data[i];
  container[size] = '\0'; // Add null terminator to container.

  const char *str_ptr = reinterpret_cast<const char *>(container);

  char *out_ptr = nullptr;

  // This fuzzer only checks that the algorithms didn't read beyond the end of
  // the string in container. Combined with sanitizers, this will check that the
  // code is not reading memory beyond what's expected. This test does not
  // effectively check the correctness of the result.
  auto volatile atof_output = __llvm_libc::atof(str_ptr);
  auto volatile strtof_output = __llvm_libc::strtof(str_ptr, &out_ptr);
  if (str_ptr + size < out_ptr)
    __builtin_trap();
  auto volatile strtod_output = __llvm_libc::strtod(str_ptr, &out_ptr);
  if (str_ptr + size < out_ptr)
    __builtin_trap();
  auto volatile strtold_output = __llvm_libc::strtold(str_ptr, &out_ptr);
  if (str_ptr + size < out_ptr)
    __builtin_trap();

  // If any of the outputs are NaN
  if (isnan(atof_output) || isnan(strtof_output) || isnan(strtod_output) ||
      isnan(strtold_output)) {
    // Then all the outputs should be NaN.
    // This is a trivial check meant to silence the "unused variable" warnings.
    if (!isnan(atof_output) || !isnan(strtof_output) || !isnan(strtod_output) ||
        !isnan(strtold_output)) {
      __builtin_trap();
    }
  }

  delete[] container;
  return 0;
}
