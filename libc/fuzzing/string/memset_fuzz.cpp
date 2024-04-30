//===-- memset_fuzz.cpp ---------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// Fuzzing test for llvm-libc memcset implementation.
///
//===----------------------------------------------------------------------===//
#include "protected_pages.h"
#include "src/string/memset.h"
#include <stddef.h> // size_t
#include <stdint.h> // uint8_t

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *data, size_t data_size) {
  static constexpr size_t MAX_SIZE = 1024;
  static ProtectedPages regions;
  static const Page write_buffer = regions.GetPageA().WithAccess(PROT_WRITE);
  // We fill 'size' with data coming from lib_fuzzer, this limits exploration to
  // 2 bytes.
  uint16_t size = 0;
  uint8_t fill_char = 0;
  if (data_size != sizeof(size) + sizeof(fill_char))
    return 0;
  __builtin_memcpy(&size, data, sizeof(size));
  __builtin_memcpy(&fill_char, data + sizeof(size), sizeof(fill_char));
  if (size >= MAX_SIZE || size >= GetPageSize())
    return 0;
  // We cross-check the function from two sources and two destinations.
  // The first of them (bottom) is always page aligned.
  // The second one (top) is not necessarily aligned.
  // Both sources and destinations are checked for out of bound accesses.
  uint8_t *destinations[2] = {write_buffer.bottom(size),
                              write_buffer.top(size)};
  for (uint8_t *dst : destinations) {
    LIBC_NAMESPACE::memset(dst, fill_char, size);
    for (size_t i = 0; i < size; ++i)
      if (dst[i] != fill_char)
        __builtin_trap();
  }
  return 0;
}
