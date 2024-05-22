//===-- Implementation header for rand utilities ----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_STDLIB_RAND_UTIL_H
#define LLVM_LIBC_SRC_STDLIB_RAND_UTIL_H

#include "src/__support/GPU/utils.h"
#include "src/__support/macros/attributes.h"

namespace LIBC_NAMESPACE {

#ifdef LIBC_TARGET_ARCH_IS_GPU
// Implement thread local storage on the GPU using local memory. Each thread
// gets its slot in the local memory array and is private to the group.
// TODO: We need to implement the 'thread_local' keyword on the GPU. This is an
// inefficient and incomplete stand-in until that is done.
template <typename T> class ThreadLocal {
private:
  static constexpr long MAX_THREADS = 1024;
  [[clang::loader_uninitialized]] static inline gpu::Local<T>
      storage[MAX_THREADS];

public:
  LIBC_INLINE operator T() const { return storage[gpu::get_thread_id()]; }
  LIBC_INLINE void operator=(const T &value) {
    storage[gpu::get_thread_id()] = value;
  }
};

extern ThreadLocal<unsigned long> rand_next;
#else
extern LIBC_THREAD_LOCAL unsigned long rand_next;
#endif

} // namespace LIBC_NAMESPACE

#endif // LLVM_LIBC_SRC_STDLIB_RAND_UTIL_H
