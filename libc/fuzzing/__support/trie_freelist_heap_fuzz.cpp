//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Fuzzing test for llvm-libc freelist-based heap implementation.
///
//===----------------------------------------------------------------------===//

#include "allocator_fuzz.h"

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *data, size_t remainder) {
  return LIBC_NAMESPACE::fuzz_one_input<
      LIBC_NAMESPACE::FreeListHeapBuffer<LIBC_NAMESPACE::heap_size>>(data,
                                                                     remainder);
}
