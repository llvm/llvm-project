//===---------- target_impl.cu - NVPTX OpenMP GPU options ------- CUDA -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Definitions of target specific functions
//
//===----------------------------------------------------------------------===//

#include "target_impl.h"
#include "common/debug.h"
#include "common/target_atomic.h"

#define __OMP_SPIN 1000
#define UNSET 0u
#define SET 1u

EXTERN void __kmpc_impl_init_lock(omp_lock_t *lock) {
  __kmpc_impl_unset_lock(lock);
}

EXTERN void __kmpc_impl_destroy_lock(omp_lock_t *lock) {
  __kmpc_impl_unset_lock(lock);
}

EXTERN void __kmpc_impl_set_lock(omp_lock_t *lock) {
  // TODO: not sure spinning is a good idea here..
  while (__kmpc_atomic_cas(lock, UNSET, SET) != UNSET) {
    clock_t start = clock();
    clock_t now;
    for (;;) {
      now = clock();
      clock_t cycles = now > start ? now - start : now + (0xffffffff - start);
      if (cycles >= __OMP_SPIN * GetBlockIdInKernel()) {
        break;
      }
    }
  } // wait for 0 to be the read value
}

EXTERN void __kmpc_impl_unset_lock(omp_lock_t *lock) {
  (void)__kmpc_atomic_exchange(lock, UNSET);
}

EXTERN int __kmpc_impl_test_lock(omp_lock_t *lock) {
  return __kmpc_atomic_add(lock, 0u);
}
