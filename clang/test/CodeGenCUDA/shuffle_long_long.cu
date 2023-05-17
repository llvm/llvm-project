// REQUIRES: nvptx-registered-target

// RUN: %clang_cc1 -triple nvptx64-nvidia-cuda -fcuda-is-device -emit-llvm -target-cpu sm_30 %s -o - | FileCheck %s --check-prefix=NO_SYNC
// RUN: %clang_cc1 -triple nvptx64-nvidia-cuda -fcuda-is-device -emit-llvm -target-cpu sm_30 -target-feature +ptx70 -DSYNC -DCUDA_VERSION=9000 %s -o - | FileCheck %s --check-prefix=SYNC

#include "Inputs/cuda.h"

__device__ void *memcpy(void *dest, const void *src, size_t n);

#define warpSize 32
#include <__clang_cuda_intrinsics.h>

__device__ void use(unsigned long long, long long);

// Test function, 4 shfl calls.
// NO_SYNC: define{{.*}} @_Z14test_long_longv
// NO_SYNC:     call noundef i64 @_Z6__shflyii(
// NO_SYNC:     call noundef i64 @_Z6__shflxii(

// SYNC: define{{.*}} @_Z14test_long_longv
// SYNC:        call noundef i64 @_Z11__shfl_syncjyii(
// SYNC:        call noundef i64 @_Z11__shfl_syncjxii(

// unsigned long long -> long long
// NO_SYNC: define{{.*}} @_Z6__shflyii
// NO_SYNC:     call noundef i64 @_Z6__shflxii(

// long long -> int + int
// NO_SYNC: define{{.*}} @_Z6__shflxii
// NO_SYNC:     call noundef i32 @_Z6__shfliii(
// NO_SYNC:     call noundef i32 @_Z6__shfliii(

// NO_SYNC: define{{.*}} @_Z6__shfliii
// NO_SYNC:   call i32 @llvm.nvvm.shfl.idx.i32

// unsigned long long -> long long
// SYNC: _Z11__shfl_syncjyii
// SYNC:     call noundef i64 @_Z11__shfl_syncjxii(

// long long -> int + int
// SYNC: define{{.*}} @_Z11__shfl_syncjxii
// SYNC:     call noundef i32 @_Z11__shfl_syncjiii(
// SYNC:     call noundef i32 @_Z11__shfl_syncjiii(

// SYNC: define{{.*}} @_Z11__shfl_syncjiii
// SYNC:      call i32 @llvm.nvvm.shfl.sync.idx.i32

__device__ void test_long_long() {
  unsigned long long ull = 13;
  long long ll = 17;
#ifndef SYNC
  ull = __shfl(ull, 7, 32);
  ll = __shfl(ll, 7, 32);
  use(ull, ll);
#else
  ull = __shfl_sync(0x11, ull, 7, 32);
  ll = __shfl_sync(0x11, ll, 7, 32);
  use(ull, ll);
#endif
}

