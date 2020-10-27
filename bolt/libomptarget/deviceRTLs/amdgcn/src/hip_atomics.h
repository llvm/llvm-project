//===---- hip_atomics.h - Declarations of hip atomic functions ---- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef OMPTARGET_AMDGCN_HIP_ATOMICS_H
#define OMPTARGET_AMDGCN_HIP_ATOMICS_H

#include "target_impl.h"

namespace {

template <typename T> DEVICE T atomicAdd(T *address, T val) {
  return __atomic_fetch_add(address, val, __ATOMIC_SEQ_CST);
}

template <typename T> DEVICE T atomicMax(T *address, T val) {
  return __atomic_fetch_max(address, val, __ATOMIC_SEQ_CST);
}

template <typename T> DEVICE T atomicExch(T *address, T val) {
  T r;
  __atomic_exchange(address, &val, &r, __ATOMIC_SEQ_CST);
  return r;
}

template <typename T> DEVICE T atomicCAS(T *address, T compare, T val) {
  (void)__atomic_compare_exchange(address, &compare, &val, false,
                                  __ATOMIC_SEQ_CST, __ATOMIC_RELAXED);
  return compare;
}

INLINE uint32_t atomicInc(uint32_t *address, uint32_t max) {
  return __builtin_amdgcn_atomic_inc32(address, max, __ATOMIC_SEQ_CST, "");
}

} // namespace
#endif
