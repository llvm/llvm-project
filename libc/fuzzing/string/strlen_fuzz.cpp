//===-- strlen_fuzz.cpp ---------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// Fuzzing test for llvm-libc strlen implementation.
///
//===----------------------------------------------------------------------===//

#include "src/string/strlen.h"
#include <cstdint>
#include <cstring>

// always null terminate the data
extern "C" size_t LLVMFuzzerMutate(uint8_t *data, size_t size, size_t max_size);
extern "C" size_t LLVMFuzzerCustomMutator(uint8_t *data, size_t size,
                                          size_t max_size, unsigned int seed) {
  size = LLVMFuzzerMutate(data, size, max_size);
  data[size - 1] = '\0';
  return size;
}

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size) {
  size_t ref = ::strlen(reinterpret_cast<const char *>(data));
  size_t impl = LIBC_NAMESPACE::strlen(reinterpret_cast<const char *>(data));
  if (ref != impl)
    __builtin_trap();
  return 0;
}
