//===-- strtointeger_fuzz.cpp ---------------------------------------------===//
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
#include "src/stdlib/atoi.h"
#include "src/stdlib/atol.h"
#include "src/stdlib/atoll.h"
#include "src/stdlib/strtol.h"
#include "src/stdlib/strtoll.h"
#include "src/stdlib/strtoul.h"
#include "src/stdlib/strtoull.h"
#include <stddef.h>
#include <stdint.h>

// This takes the randomized bytes in data and interprets the first byte as the
// base for the string to integer conversion and the rest of them as a string to
// be passed to the string to integer conversion.
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size) {
  size_t container_size = 0;
  if (size == 0) {
    container_size = 1;
  } else {
    container_size = size;
  }
  uint8_t *container = new uint8_t[container_size];
  if (!container)
    __builtin_trap();

  int base = 0;
  if (size > 0) {
    base = data[0] % 36;
    base = base + ((base == 0) ? 0 : 1);
  }
  for (size_t i = 1; i < size; ++i) {
    container[i - 1] = data[i];
  }

  container[container_size - 1] = '\0'; // Add null terminator to container.

  const char *str_ptr = reinterpret_cast<const char *>(container);

  char *out_ptr = nullptr;

  auto volatile atoi_output = __llvm_libc::atoi(str_ptr);
  auto volatile atol_output = __llvm_libc::atol(str_ptr);
  auto volatile atoll_output = __llvm_libc::atoll(str_ptr);
  auto volatile strtol_output = __llvm_libc::strtol(str_ptr, &out_ptr, base);
  if (str_ptr + container_size - 1 < out_ptr)
    __builtin_trap();
  auto volatile strtoll_output = __llvm_libc::strtoll(str_ptr, &out_ptr, base);
  if (str_ptr + container_size - 1 < out_ptr)
    __builtin_trap();
  auto volatile strtoul_output = __llvm_libc::strtoul(str_ptr, &out_ptr, base);
  if (str_ptr + container_size - 1 < out_ptr)
    __builtin_trap();
  auto volatile strtoull_output =
      __llvm_libc::strtoull(str_ptr, &out_ptr, base);
  if (str_ptr + container_size - 1 < out_ptr)
    __builtin_trap();

  // If atoi is non-zero and the base is at least 10
  if (atoi_output != 0 && base >= 10) {
    // Then all of the other functions should output non-zero values as well.
    // This is a trivial check meant to silence the "unused variable" warnings.
    if (atol_output == 0 || atoll_output == 0 || strtol_output == 0 ||
        strtoll_output == 0 || strtoul_output == 0 || strtoull_output == 0) {
      __builtin_trap();
    }
  }

  delete[] container;
  return 0;
}
