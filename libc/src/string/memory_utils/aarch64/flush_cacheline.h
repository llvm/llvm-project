//===-- Flush Cacheline for AArch64 -----------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#ifndef LLVM_LIBC_SRC_STRING_MEMORY_UTILS_AARCH64_FLUSH_CACHELINE_H
#define LLVM_LIBC_SRC_STRING_MEMORY_UTILS_AARCH64_FLUSH_CACHELINE_H

#include "src/__support/common.h"
#include <stddef.h> // size_t
namespace LIBC_NAMESPACE {

LIBC_INLINE size_t cacheline_size() {
  // Use the same way as in compiler-rt
  size_t ctr_el0;
  asm volatile("mrs %0, ctr_el0" : "=r"(ctr_el0));
  return 4 << ((ctr_el0 >> 16) & 15);
}

LIBC_INLINE void flush_cacheline_async(volatile char *addr) {
  // flush to external memory and invalidate the cache line
  asm volatile("dc civac, %0" : : "r"(addr) : "memory");
}

} // namespace LIBC_NAMESPACE

#endif // LLVM_LIBC_SRC_STRING_MEMORY_UTILS_AARCH64_FLUSH_CACHELINE_H
