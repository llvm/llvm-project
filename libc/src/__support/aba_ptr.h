//===-- Transactional Ptr for ABA prevention --------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC___SUPPORT_ABA_PTR_H
#define LLVM_LIBC_SRC___SUPPORT_ABA_PTR_H

#include "hdr/types/size_t.h"
#include "src/__support/CPP/atomic.h"
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
        cpp::AtomicRef<typename Impl::Atomic> ref(impl.atomic);
        typename Impl::Atomic snapshot, next;
        snapshot = ref.load(cpp::MemoryOrder::RELAXED);
        next.ptr = op(snapshot.ptr);
        // Wrapping add for unsigned integers.
        next.tag = snapshot.tag + 1;
        // Redo transaction can be costly, so we use strong version.
        if (ref.compare_exchange_strong(snapshot, next,
                                        cpp::MemoryOrder::ACQ_REL,
                                        cpp::MemoryOrder::RELAXED))
          return;
      }
    } else {
      // Acquire the lock.
      cpp::AtomicRef<bool> ref(impl.mutex.locked);
      while (ref.exchange(true, cpp::MemoryOrder::ACQUIRE))
        while (ref.load(cpp::MemoryOrder::RELAXED))
          LIBC_NAMESPACE::sleep_briefly();

      impl.mutex.ptr = op(impl.mutex.ptr);
      // Release the lock.
      ref.store(false, cpp::MemoryOrder::RELEASE);
    }
  }

  LIBC_INLINE T *get() const {
    if constexpr (IsAtomic) {
      // Weak micro-architectures typically regards simultaneous partial word
      // loading and full word loading as a race condition. While there are
      // implementations that uses racy read anyway, we still load the whole
      // word to avoid any complications.
      typename Impl::Atomic snapshot;
      cpp::AtomicRef<typename Impl::Atomic> ref(impl.atomic);
      snapshot = ref.load(cpp::MemoryOrder::RELAXED);
      return snapshot.ptr;
    } else {
      return impl.mutex.ptr;
    }
  }
};

template <class T> using AbaPtr = AbaPtrImpl<T, LIBC_ABA_PTR_IS_ATOMIC>;
} // namespace LIBC_NAMESPACE_DECL

#endif
