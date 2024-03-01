//===-- Flush Cacheline for x86_64 ------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#ifndef LLVM_LIBC_SRC_STRING_MEMORY_UTILS_X86_64_FLUSH_CACHELINE_H
#define LLVM_LIBC_SRC_STRING_MEMORY_UTILS_X86_64_FLUSH_CACHELINE_H

#include "src/__support/common.h"
#include <stddef.h> // size_t
namespace LIBC_NAMESPACE {

LIBC_INLINE constexpr size_t cacheline_size() { return 64; }

LIBC_INLINE void flush_cacheline_async(volatile char *addr) {
#if defined(LIBC_TARGET_CPU_HAS_CLFLUSHOPT)
  asm volatile("clflushopt %0" : "+m"(*addr)::"memory");
#else
  __builtin_ia32_clflush(const_cast<const char *>(addr));
#endif
}

} // namespace LIBC_NAMESPACE

#endif // LLVM_LIBC_SRC_STRING_MEMORY_UTILS_X86_64_FLUSH_CACHELINE_H
