// RUN: %clang_cc1 -triple nvptx64-unknown-unknown -target-cpu sm_60 \
// RUN:            -fcuda-is-device -target-feature +ptx60 \
// RUN:            -S -emit-llvm -o - -x cuda %s \
// RUN:   | FileCheck -check-prefix=CHECK %s
// RUN: %clang_cc1 -triple nvptx-unknown-unknown -target-cpu sm_60 \
// RUN:   -fcuda-is-device -S -o /dev/null -x cuda -verify %s

#define __device__ __attribute__((device))
#define __global__ __attribute__((global))
#define __shared__ __attribute__((shared))
#define __constant__ __attribute__((constant))

// CHECK-LABEL: nvvm_shfl_sync
__device__ void nvvm_shfl_sync(unsigned mask, int i, float f, int a, int b) {
  // CHECK: call i32 @llvm.nvvm.shfl.sync.down.i32(i32 {{%[0-9]+}}, i32
  // expected-error@+1 {{'__nvvm_shfl_sync_down_i32' needs target feature ptx60}}
  __nvvm_shfl_sync_down_i32(mask, i, a, b);
  // CHECK: call float @llvm.nvvm.shfl.sync.down.f32(i32 {{%[0-9]+}}, float
  // expected-error@+1 {{'__nvvm_shfl_sync_down_f32' needs target feature ptx60}}
  __nvvm_shfl_sync_down_f32(mask, f, a, b);
  // CHECK: call i32 @llvm.nvvm.shfl.sync.up.i32(i32 {{%[0-9]+}}, i32
  // expected-error@+1 {{'__nvvm_shfl_sync_up_i32' needs target feature ptx60}}
  __nvvm_shfl_sync_up_i32(mask, i, a, b);
  // CHECK: call float @llvm.nvvm.shfl.sync.up.f32(i32 {{%[0-9]+}}, float
  // expected-error@+1 {{'__nvvm_shfl_sync_up_f32' needs target feature ptx60}}
  __nvvm_shfl_sync_up_f32(mask, f, a, b);
  // CHECK: call i32 @llvm.nvvm.shfl.sync.bfly.i32(i32 {{%[0-9]+}}, i32
  // expected-error@+1 {{'__nvvm_shfl_sync_bfly_i32' needs target feature ptx60}}
  __nvvm_shfl_sync_bfly_i32(mask, i, a, b);
  // CHECK: call float @llvm.nvvm.shfl.sync.bfly.f32(i32 {{%[0-9]+}}, float
  // expected-error@+1 {{'__nvvm_shfl_sync_bfly_f32' needs target feature ptx60}}
  __nvvm_shfl_sync_bfly_f32(mask, f, a, b);
  // CHECK: call i32 @llvm.nvvm.shfl.sync.idx.i32(i32 {{%[0-9]+}}, i32
  // expected-error@+1 {{'__nvvm_shfl_sync_idx_i32' needs target feature ptx60}}
  __nvvm_shfl_sync_idx_i32(mask, i, a, b);
  // CHECK: call float @llvm.nvvm.shfl.sync.idx.f32(i32 {{%[0-9]+}}, float
  // expected-error@+1 {{'__nvvm_shfl_sync_idx_f32' needs target feature ptx60}}
  __nvvm_shfl_sync_idx_f32(mask, f, a, b);
  // CHECK: ret void
}
