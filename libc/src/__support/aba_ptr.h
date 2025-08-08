//===-- Transactional Ptr for ABA prevention --------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC___SUPPORT_TAGGED_POINTER_H
#define LLVM_LIBC_SRC___SUPPORT_TAGGED_POINTER_H

#include "hdr/types/size_t.h"
#include "src/__support/common.h"
#include "src/__support/threads/sleep.h"

#ifdef __GCC_HAVE_SYNC_COMPARE_AND_SWAP_16
#define LIBC_ABA_PTR_IS_ATOMIC true
#else
#define LIBC_ABA_PTR_IS_ATOMIC false
#endif

namespace LIBC_NAMESPACE_DECL {

template <class T, bool IsAtomic> struct AbaPtrImpl {
  union Impl {
    struct alignas(2 * alignof(void *)) Atomic {
      T *ptr;
      size_t tag;
    } atomic;
    struct Mutex {
      T *ptr;
      bool locked;
    } mutex;
  } impl;

  LIBC_INLINE constexpr AbaPtrImpl(T *ptr)
      : impl(IsAtomic ? Impl{.atomic{ptr, 0}} : Impl{.mutex{ptr, false}}) {}

  /// User must guarantee that operation is redoable.
  template <class Op> LIBC_INLINE void transaction(Op &&op) {
    if constexpr (IsAtomic) {
      for (;;) {
        typename Impl::Atomic snapshot, next;
        __atomic_load(&impl.atomic, &snapshot, __ATOMIC_RELAXED);
        next.ptr = op(snapshot.ptr);
        // Wrapping add for unsigned integers.
        next.tag = snapshot.tag + 1;
        if (__atomic_compare_exchange(&impl.atomic, &snapshot, &next, true,
                                      __ATOMIC_ACQ_REL, __ATOMIC_RELAXED))
          return;
      }
    } else {
      // Acquire the lock.
      while (__atomic_exchange_n(&impl.mutex.locked, true, __ATOMIC_ACQUIRE))
        while (__atomic_load_n(&impl.mutex.locked, __ATOMIC_RELAXED))
          LIBC_NAMESPACE::sleep_briefly();

      impl.mutex.ptr = op(impl.mutex.ptr);
      // Release the lock.
      __atomic_store_n(&impl.mutex.locked, false, __ATOMIC_RELEASE);
    }
  }

  LIBC_INLINE T *get() const {
    if constexpr (IsAtomic) {
      // Weak micro-architectures typically regards simultaneous partial word
      // loading and full word loading as a race condition. While there are
      // implementations that uses racy read anyway, we still load the whole
      // word to avoid any complications.
      typename Impl::Atomic snapshot;
      __atomic_load(&impl.atomic, &snapshot, __ATOMIC_RELAXED);
      return snapshot.ptr;
    } else {
      return impl.mutex.ptr;
    }
  }
};

template <class T> using AbaPtr = AbaPtrImpl<T, LIBC_ABA_PTR_IS_ATOMIC>;
} // namespace LIBC_NAMESPACE_DECL

#undef LIBC_ABA_PTR_IS_ATOMIC
#endif
