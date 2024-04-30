//===-- memcpy_fuzz.cpp ---------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// Fuzzing test for llvm-libc memcpy implementation.
///
//===----------------------------------------------------------------------===//
#include "protected_pages.h"
#include "src/string/memcpy.h"
#include <stddef.h> // size_t
#include <stdint.h> // uint8_t
#include <stdlib.h> // rand

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *data, size_t data_size) {
  static constexpr size_t MAX_SIZE = 1024;
  static ProtectedPages regions;
  static const Page write_buffer = regions.GetPageA().WithAccess(PROT_WRITE);
  static const Page read_buffer = [&]() {
    // We fetch page B in write mode.
    auto region = regions.GetPageB().WithAccess(PROT_WRITE);
    // And fill it with random numbers.
    for (size_t i = 0; i < region.page_size; ++i)
      region.page_ptr[i] = rand();
    // Then return it in read mode.
    return region.WithAccess(PROT_READ);
  }();
  // We fill 'size' with data coming from lib_fuzzer, this limits exploration to
  // 2 bytes.
  uint16_t size = 0;
  if (data_size != sizeof(size))
    return 0;
  __builtin_memcpy(&size, data, sizeof(size));
  if (size >= MAX_SIZE || size >= GetPageSize())
    return 0;
  // We cross-check the function from two sources and two destinations.
  // The first of them (bottom) is always page aligned.
  // The second one (top) is not necessarily aligned.
  // Both sources and destinations are checked for out of bound accesses.
  const uint8_t *sources[2] = {read_buffer.bottom(size), read_buffer.top(size)};
  uint8_t *destinations[2] = {write_buffer.bottom(size),
                              write_buffer.top(size)};
  for (const uint8_t *src : sources) {
    for (uint8_t *dst : destinations) {
      LIBC_NAMESPACE::memcpy(dst, src, size);
      for (size_t i = 0; i < size; ++i)
        if (src[i] != dst[i])
          __builtin_trap();
    }
  }
  return 0;
}
