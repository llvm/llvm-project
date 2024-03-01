//===-- Dispatch cache flushing -------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_STRING_MEMORY_UTILS_FLUSH_CACHE_H
#define LLVM_LIBC_SRC_STRING_MEMORY_UTILS_FLUSH_CACHE_H

#include "src/__support/CPP/atomic.h"
#include "src/__support/common.h"
#include "src/__support/macros/properties/architectures.h" // LIBC_TARGET_ARCH_IS_

#include <stddef.h> // size_t
#include <stdint.h> // uintptr_t

#ifdef LIBC_TARGET_ARCH_IS_X86
#include "src/string/memory_utils/x86_64/flush_cacheline.h"
#define LIBC_HAS_FLUSH_CACHELINE_ASYNC 1
#elif defined(LIBC_TARGET_ARCH_IS_AARCH64)
#include "src/string/memory_utils/aarch64/flush_cacheline.h"
#define LIBC_HAS_FLUSH_CACHELINE_ASYNC 1
#else
#define LIBC_HAS_FLUSH_CACHELINE_ASYNC 0
#endif

namespace LIBC_NAMESPACE {

LIBC_INLINE void flush_cache(volatile void *start, size_t size) {
#if LIBC_HAS_FLUSH_CACHELINE_ASYNC
  size_t line_size = cacheline_size();
  uintptr_t addr = reinterpret_cast<uintptr_t>(start);
  uintptr_t offset = addr % line_size;
  // shift start to the left and align size to the right
  // we want to cover the whole range of memory that needs to be flushed
  size += offset;
  size += line_size - (size % line_size);
  addr -= offset;
  // flush cache line async may be reordered. We need to put barriers.
  cpp::atomic_thread_fence(cpp::MemoryOrder::SEQ_CST);
  for (size_t i = 0; i < size; i += line_size)
    flush_cacheline_async(reinterpret_cast<volatile char *>(addr + i));
  cpp::atomic_thread_fence(cpp::MemoryOrder::SEQ_CST);
#else
  // we do not have specific instructions to flush the cache
  // fallback to use a full memory barrier instead.
  // Notice, however, memory fence might not flush the cache on many
  // architectures.
  cpp::atomic_thread_fence(cpp::MemoryOrder::SEQ_CST);
#endif
}

} // namespace LIBC_NAMESPACE
#undef LIBC_HAS_FLUSH_CACHELINE_ASYNC
#endif // LLVM_LIBC_SRC_STRING_MEMORY_UTILS_FLUSH_CACHE_H
