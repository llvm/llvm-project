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

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size) {
  uint8_t *container = new uint8_t[size + 1];
  if (!container)
    __builtin_trap();
  size_t i;

  for (i = 0; i < size; ++i)
    container[i] = data[i];
  container[size] = '\0'; // Add null terminator to container.

  StringParserOutputDiff<double>(&LIBC_NAMESPACE::atof, &::atof, container,
                                 size);
  delete[] container;
  return 0;
}
