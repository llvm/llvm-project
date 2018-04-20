//===-- sanitizer_atomic.h --------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file is a part of ThreadSanitizer/AddressSanitizer runtime.
//
//===----------------------------------------------------------------------===//

#ifndef SANITIZER_ATOMIC_H
#define SANITIZER_ATOMIC_H

#include "sanitizer_internal_defs.h"

namespace __sanitizer {

enum memory_order {
  memory_order_relaxed = 1 << 0,
  memory_order_consume = 1 << 1,
  memory_order_acquire = 1 << 2,
  memory_order_release = 1 << 3,
  memory_order_acq_rel = 1 << 4,
  memory_order_seq_cst = 1 << 5
};

template<typename T>
struct atomic {
  typedef T Type;
  volatile Type ALIGNED(sizeof(Type)) val_dont_use;
};

typedef atomic<u8> atomic_uint8_t;
typedef atomic<u16> atomic_uint16_t;
typedef atomic<s32> atomic_sint32_t;
typedef atomic<u32> atomic_uint32_t;
typedef atomic<u64> atomic_uint64_t;
typedef atomic<uptr> atomic_uintptr_t;

}  // namespace __sanitizer

#if defined(__clang__) || defined(__GNUC__)
# include "sanitizer_atomic_clang.h"
#elif defined(_MSC_VER)
# include "sanitizer_atomic_msvc.h"
#else
# error "Unsupported compiler"
#endif

namespace __sanitizer {

// Clutter-reducing helpers.

template<typename T>
INLINE typename T::Type atomic_load_relaxed(const volatile T *a) {
  return atomic_load(a, memory_order_relaxed);
}

template<typename T>
INLINE void atomic_store_relaxed(volatile T *a, typename T::Type v) {
  atomic_store(a, v, memory_order_relaxed);
}

}  // namespace __sanitizer

#endif  // SANITIZER_ATOMIC_H
