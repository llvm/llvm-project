//===-- sanitizer_atomic_clang_other.h --------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file is a part of ThreadSanitizer/AddressSanitizer runtime.
// Not intended for direct inclusion. Include sanitizer_atomic.h.
//
//===----------------------------------------------------------------------===//

#ifndef SANITIZER_ATOMIC_CLANG_OTHER_H
#define SANITIZER_ATOMIC_CLANG_OTHER_H

namespace __sanitizer {

// MIPS32 does not support atomic > 4 bytes. To address this lack of
// functionality, the sanitizer library provides helper methods which use an
// internal spin lock mechanism to emulate atomic oprations when the size is
// 8 bytes.
#if defined(_MIPS_SIM) && _MIPS_SIM == _ABIO32
static void __spin_lock(volatile int *lock) {
  while (__sync_lock_test_and_set(lock, 1))
    while (*lock) {
    }
}

static void __spin_unlock(volatile int *lock) { __sync_lock_release(lock); }


// Make sure the lock is on its own cache line to prevent false sharing.
// Put it inside a struct that is aligned and padded to the typical MIPS
// cacheline which is 32 bytes.
static struct {
  int lock;
  char pad[32 - sizeof(int)];
} __attribute__((aligned(32))) lock = {0};

template <class T>
T __mips_sync_fetch_and_add(volatile T *ptr, T val) {
  T ret;

  __spin_lock(&lock.lock);

  ret = *ptr;
  *ptr = ret + val;

  __spin_unlock(&lock.lock);

  return ret;
}

template <class T>
T __mips_sync_val_compare_and_swap(volatile T *ptr, T oldval, T newval) {
  T ret;
  __spin_lock(&lock.lock);

  ret = *ptr;
  if (ret == oldval) *ptr = newval;

  __spin_unlock(&lock.lock);

  return ret;
}
#endif

INLINE void proc_yield(int cnt) {
  __asm__ __volatile__("" ::: "memory");
}

template<typename T>
INLINE typename T::Type atomic_load(
    const volatile T *a, memory_order mo) {
  DCHECK(mo & (memory_order_relaxed | memory_order_consume
      | memory_order_acquire | memory_order_seq_cst));
  DCHECK(!((uptr)a % sizeof(*a)));
  typename T::Type v;

  if (sizeof(*a) < 8 || sizeof(void*) == 8) {
    // Assume that aligned loads are atomic.
    if (mo == memory_order_relaxed) {
      v = a->val_dont_use;
    } else if (mo == memory_order_consume) {
      // Assume that processor respects data dependencies
      // (and that compiler won't break them).
      __asm__ __volatile__("" ::: "memory");
      v = a->val_dont_use;
      __asm__ __volatile__("" ::: "memory");
    } else if (mo == memory_order_acquire) {
      __asm__ __volatile__("" ::: "memory");
      v = a->val_dont_use;
      __sync_synchronize();
    } else {  // seq_cst
      // E.g. on POWER we need a hw fence even before the store.
      __sync_synchronize();
      v = a->val_dont_use;
      __sync_synchronize();
    }
  } else {
    // 64-bit load on 32-bit platform.
    // Gross, but simple and reliable.
    // Assume that it is not in read-only memory.
#if defined(_MIPS_SIM) && _MIPS_SIM == _ABIO32
    typename T::Type volatile *val_ptr =
        const_cast<typename T::Type volatile *>(&a->val_dont_use);
    v = __mips_sync_fetch_and_add<u64>(
        reinterpret_cast<u64 volatile *>(val_ptr), 0);
#else
    v = __sync_fetch_and_add(
        const_cast<typename T::Type volatile *>(&a->val_dont_use), 0);
#endif
  }
  return v;
}

template<typename T>
INLINE void atomic_store(volatile T *a, typename T::Type v, memory_order mo) {
  DCHECK(mo & (memory_order_relaxed | memory_order_release
      | memory_order_seq_cst));
  DCHECK(!((uptr)a % sizeof(*a)));

  if (sizeof(*a) < 8 || sizeof(void*) == 8) {
    // Assume that aligned loads are atomic.
    if (mo == memory_order_relaxed) {
      a->val_dont_use = v;
    } else if (mo == memory_order_release) {
      __sync_synchronize();
      a->val_dont_use = v;
      __asm__ __volatile__("" ::: "memory");
    } else {  // seq_cst
      __sync_synchronize();
      a->val_dont_use = v;
      __sync_synchronize();
    }
  } else {
    // 64-bit store on 32-bit platform.
    // Gross, but simple and reliable.
    typename T::Type cmp = a->val_dont_use;
    typename T::Type cur;
    for (;;) {
#if defined(_MIPS_SIM) && _MIPS_SIM == _ABIO32
      typename T::Type volatile *val_ptr =
          const_cast<typename T::Type volatile *>(&a->val_dont_use);
      cur = __mips_sync_val_compare_and_swap<u64>(
          reinterpret_cast<u64 volatile *>(val_ptr), reinterpret_cast<u64> cmp,
          reinterpret_cast<u64> v);
#else
      cur = __sync_val_compare_and_swap(&a->val_dont_use, cmp, v);
#endif
      if (cmp == v)
        break;
      cmp = cur;
    }
  }
}

}  // namespace __sanitizer

#endif  // #ifndef SANITIZER_ATOMIC_CLANG_OTHER_H
