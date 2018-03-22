// RUN: %clang_cc1 -triple nvptx64-unknown-unknown -target-cpu sm_70 \
// RUN:            -fcuda-is-device -target-feature +ptx60 \
// RUN:            -S -emit-llvm -o - -x cuda %s \
// RUN:   | FileCheck -check-prefix=CHECK %s
// RUN: %clang_cc1 -triple nvptx-unknown-unknown -target-cpu sm_60 \
// RUN:   -fcuda-is-device -S -o /dev/null -x cuda -verify %s

#if !defined(CUDA_VERSION)
#define __device__ __attribute__((device))
#define __global__ __attribute__((global))
#define __shared__ __attribute__((shared))
#define __constant__ __attribute__((constant))

typedef unsigned long long uint64_t;
#endif
// We have to keep all builtins that depend on particular target feature in the
// same function, because the codegen will stop after the very first function
// that encounters an error, so -verify will not be able to find errors in
// subsequent functions.

// CHECK-LABEL: nvvm_wmma
__device__ void nvvm_wmma(int *src, int *dst,
                          float *fsrc, float *fdst,
                          int ldm) {
  // CHECK: call {{.*}} @llvm.nvvm.wmma.m16n16k16.load.a.row.stride.f16
  // expected-error@+1 {{'__hmma_m16n16k16_ld_a' needs target feature ptx60}}
  __hmma_m16n16k16_ld_a(dst, src, ldm, 0);
  // CHECK: call {{.*}} @llvm.nvvm.wmma.m16n16k16.load.a.col.stride.f16
  // expected-error@+1 {{'__hmma_m16n16k16_ld_a' needs target feature ptx60}}
  __hmma_m16n16k16_ld_a(dst, src+1, ldm, 1);

  // CHECK: call {{.*}} @llvm.nvvm.wmma.m16n16k16.load.b.row.stride.f16
  // expected-error@+1 {{'__hmma_m16n16k16_ld_b' needs target feature ptx60}}
  __hmma_m16n16k16_ld_b(dst, src, ldm, 0);
  // CHECK: call {{.*}} @llvm.nvvm.wmma.m16n16k16.load.b.col.stride.f16
  // expected-error@+1 {{'__hmma_m16n16k16_ld_b' needs target feature ptx60}}
  __hmma_m16n16k16_ld_b(dst, src+2, ldm, 1);

  // CHECK: call {{.*}} @llvm.nvvm.wmma.m16n16k16.load.c.row.stride.f16
  // expected-error@+1 {{'__hmma_m16n16k16_ld_c_f16' needs target feature ptx60}}
  __hmma_m16n16k16_ld_c_f16(dst, src, ldm, 0);
  // CHECK: call {{.*}} @llvm.nvvm.wmma.m16n16k16.load.c.col.stride.f16
  // expected-error@+1 {{'__hmma_m16n16k16_ld_c_f16' needs target feature ptx60}}
  __hmma_m16n16k16_ld_c_f16(dst, src, ldm, 1);

  // CHECK: call {{.*}} @llvm.nvvm.wmma.m16n16k16.load.c.row.stride.f32
  // expected-error@+1 {{'__hmma_m16n16k16_ld_c_f32' needs target feature ptx60}}
  __hmma_m16n16k16_ld_c_f32(fdst, fsrc, ldm, 0);
  // CHECK: call {{.*}} @llvm.nvvm.wmma.m16n16k16.load.c.col.stride.f32
  // expected-error@+1 {{'__hmma_m16n16k16_ld_c_f32' needs target feature ptx60}}
  __hmma_m16n16k16_ld_c_f32(fdst, fsrc, ldm, 1);

  // CHECK: call {{.*}} @llvm.nvvm.wmma.m16n16k16.store.d.row.stride.f16
  // expected-error@+1 {{'__hmma_m16n16k16_st_c_f16' needs target feature ptx60}}
  __hmma_m16n16k16_st_c_f16(dst, src, ldm, 0);
  // CHECK: call {{.*}} @llvm.nvvm.wmma.m16n16k16.store.d.col.stride.f16
  // expected-error@+1 {{'__hmma_m16n16k16_st_c_f16' needs target feature ptx60}}
  __hmma_m16n16k16_st_c_f16(dst, src, ldm, 1);

  // CHECK: call {{.*}} @llvm.nvvm.wmma.m16n16k16.store.d.row.stride.f32
  // expected-error@+1 {{'__hmma_m16n16k16_st_c_f32' needs target feature ptx60}}
  __hmma_m16n16k16_st_c_f32(fdst, fsrc, ldm, 0);
  // CHECK: call {{.*}} @llvm.nvvm.wmma.m16n16k16.store.d.col.stride.f32
  // expected-error@+1 {{'__hmma_m16n16k16_st_c_f32' needs target feature ptx60}}
  __hmma_m16n16k16_st_c_f32(fdst, fsrc, ldm, 1);

  // CHECK: call {{.*}} @llvm.nvvm.wmma.m16n16k16.mma.row.row.f16.f16
  // expected-error@+1 {{'__hmma_m16n16k16_mma_f16f16' needs target feature ptx60}}
  __hmma_m16n16k16_mma_f16f16(dst, src, src, src, 0, 0);
  // CHECK: call {{.*}} @llvm.nvvm.wmma.m16n16k16.mma.row.row.f16.f16.satfinite
  // expected-error@+1 {{'__hmma_m16n16k16_mma_f16f16' needs target feature ptx60}}
  __hmma_m16n16k16_mma_f16f16(dst, src, src, src, 0, 1);
  // CHECK: call {{.*}} @llvm.nvvm.wmma.m16n16k16.mma.row.col.f16.f16
  // expected-error@+1 {{'__hmma_m16n16k16_mma_f16f16' needs target feature ptx60}}
  __hmma_m16n16k16_mma_f16f16(dst, src, src, src, 1, 0);
  // CHECK: call {{.*}} @llvm.nvvm.wmma.m16n16k16.mma.row.col.f16.f16.satfinite
  // expected-error@+1 {{'__hmma_m16n16k16_mma_f16f16' needs target feature ptx60}}
  __hmma_m16n16k16_mma_f16f16(dst, src, src, src, 1, 1);
  // CHECK: call {{.*}} @llvm.nvvm.wmma.m16n16k16.mma.col.row.f16.f16
  // expected-error@+1 {{'__hmma_m16n16k16_mma_f16f16' needs target feature ptx60}}
  __hmma_m16n16k16_mma_f16f16(dst, src, src, src, 2, 0);
  // CHECK: call {{.*}} @llvm.nvvm.wmma.m16n16k16.mma.col.row.f16.f16.satfinite
  // expected-error@+1 {{'__hmma_m16n16k16_mma_f16f16' needs target feature ptx60}}
  __hmma_m16n16k16_mma_f16f16(dst, src, src, src, 2, 1);
  // CHECK: call {{.*}} @llvm.nvvm.wmma.m16n16k16.mma.col.col.f16.f16
  // expected-error@+1 {{'__hmma_m16n16k16_mma_f16f16' needs target feature ptx60}}
  __hmma_m16n16k16_mma_f16f16(dst, src, src, src, 3, 0);
  // CHECK: call {{.*}} @llvm.nvvm.wmma.m16n16k16.mma.col.col.f16.f16.satfinite
  // expected-error@+1 {{'__hmma_m16n16k16_mma_f16f16' needs target feature ptx60}}
  __hmma_m16n16k16_mma_f16f16(dst, src, src, src, 3, 1);

  // CHECK: call {{.*}} @llvm.nvvm.wmma.m16n16k16.mma.row.row.f16.f32
  // expected-error@+1 {{'__hmma_m16n16k16_mma_f16f32' needs target feature ptx60}}
  __hmma_m16n16k16_mma_f16f32(dst, src, src, fsrc, 0, 0);
  // CHECK: call {{.*}} @llvm.nvvm.wmma.m16n16k16.mma.row.row.f16.f32.satfinite
  // expected-error@+1 {{'__hmma_m16n16k16_mma_f16f32' needs target feature ptx60}}
  __hmma_m16n16k16_mma_f16f32(dst, src, src, fsrc, 0, 1);
  // CHECK: call {{.*}} @llvm.nvvm.wmma.m16n16k16.mma.row.col.f16.f32
  // expected-error@+1 {{'__hmma_m16n16k16_mma_f16f32' needs target feature ptx60}}
  __hmma_m16n16k16_mma_f16f32(dst, src, src, fsrc, 1, 0);
  // CHECK: call {{.*}} @llvm.nvvm.wmma.m16n16k16.mma.row.col.f16.f32.satfinite
  // expected-error@+1 {{'__hmma_m16n16k16_mma_f16f32' needs target feature ptx60}}
  __hmma_m16n16k16_mma_f16f32(dst, src, src, fsrc, 1, 1);
  // CHECK: call {{.*}} @llvm.nvvm.wmma.m16n16k16.mma.col.row.f16.f32
  // expected-error@+1 {{'__hmma_m16n16k16_mma_f16f32' needs target feature ptx60}}
  __hmma_m16n16k16_mma_f16f32(dst, src, src, fsrc, 2, 0);
  // CHECK: call {{.*}} @llvm.nvvm.wmma.m16n16k16.mma.col.row.f16.f32.satfinite
  // expected-error@+1 {{'__hmma_m16n16k16_mma_f16f32' needs target feature ptx60}}
  __hmma_m16n16k16_mma_f16f32(dst, src, src, fsrc, 2, 1);
  // CHECK: call {{.*}} @llvm.nvvm.wmma.m16n16k16.mma.col.col.f16.f32
  // expected-error@+1 {{'__hmma_m16n16k16_mma_f16f32' needs target feature ptx60}}
  __hmma_m16n16k16_mma_f16f32(dst, src, src, fsrc, 3, 0);
  // CHECK: call {{.*}} @llvm.nvvm.wmma.m16n16k16.mma.col.col.f16.f32.satfinite
  // expected-error@+1 {{'__hmma_m16n16k16_mma_f16f32' needs target feature ptx60}}
  __hmma_m16n16k16_mma_f16f32(dst, src, src, fsrc, 3, 1);

  // CHECK: call {{.*}} @llvm.nvvm.wmma.m16n16k16.mma.row.row.f32.f16
  // expected-error@+1 {{'__hmma_m16n16k16_mma_f32f16' needs target feature ptx60}}
  __hmma_m16n16k16_mma_f32f16(fdst, src, src, src, 0, 0);
  // CHECK: call {{.*}} @llvm.nvvm.wmma.m16n16k16.mma.row.row.f32.f16.satfinite
  // expected-error@+1 {{'__hmma_m16n16k16_mma_f32f16' needs target feature ptx60}}
  __hmma_m16n16k16_mma_f32f16(fdst, src, src, src, 0, 1);
  // CHECK: call {{.*}} @llvm.nvvm.wmma.m16n16k16.mma.row.col.f32.f16
  // expected-error@+1 {{'__hmma_m16n16k16_mma_f32f16' needs target feature ptx60}}
  __hmma_m16n16k16_mma_f32f16(fdst, src, src, src, 1, 0);
  // CHECK: call {{.*}} @llvm.nvvm.wmma.m16n16k16.mma.row.col.f32.f16.satfinite
  // expected-error@+1 {{'__hmma_m16n16k16_mma_f32f16' needs target feature ptx60}}
  __hmma_m16n16k16_mma_f32f16(fdst, src, src, src, 1, 1);
  // CHECK: call {{.*}} @llvm.nvvm.wmma.m16n16k16.mma.col.row.f32.f16
  // expected-error@+1 {{'__hmma_m16n16k16_mma_f32f16' needs target feature ptx60}}
  __hmma_m16n16k16_mma_f32f16(fdst, src, src, src, 2, 0);
  // CHECK: call {{.*}} @llvm.nvvm.wmma.m16n16k16.mma.col.row.f32.f16.satfinite
  // expected-error@+1 {{'__hmma_m16n16k16_mma_f32f16' needs target feature ptx60}}
  __hmma_m16n16k16_mma_f32f16(fdst, src, src, src, 2, 1);
  // CHECK: call {{.*}} @llvm.nvvm.wmma.m16n16k16.mma.col.col.f32.f16
  // expected-error@+1 {{'__hmma_m16n16k16_mma_f32f16' needs target feature ptx60}}
  __hmma_m16n16k16_mma_f32f16(fdst, src, src, src, 3, 0);
  // CHECK: call {{.*}} @llvm.nvvm.wmma.m16n16k16.mma.col.col.f32.f16.satfinite
  // expected-error@+1 {{'__hmma_m16n16k16_mma_f32f16' needs target feature ptx60}}
  __hmma_m16n16k16_mma_f32f16(fdst, src, src, src, 3, 1);

  // CHECK: call {{.*}} @llvm.nvvm.wmma.m16n16k16.mma.row.row.f32.f32
  // expected-error@+1 {{'__hmma_m16n16k16_mma_f32f32' needs target feature ptx60}}
  __hmma_m16n16k16_mma_f32f32(fdst, src, src, fsrc, 0, 0);
  // CHECK: call {{.*}} @llvm.nvvm.wmma.m16n16k16.mma.row.row.f32.f32.satfinite
  // expected-error@+1 {{'__hmma_m16n16k16_mma_f32f32' needs target feature ptx60}}
  __hmma_m16n16k16_mma_f32f32(fdst, src, src, fsrc, 0, 1);
  // CHECK: call {{.*}} @llvm.nvvm.wmma.m16n16k16.mma.row.col.f32.f32
  // expected-error@+1 {{'__hmma_m16n16k16_mma_f32f32' needs target feature ptx60}}
  __hmma_m16n16k16_mma_f32f32(fdst, src, src, fsrc, 1, 0);
  // CHECK: call {{.*}} @llvm.nvvm.wmma.m16n16k16.mma.row.col.f32.f32.satfinite
  // expected-error@+1 {{'__hmma_m16n16k16_mma_f32f32' needs target feature ptx60}}
  __hmma_m16n16k16_mma_f32f32(fdst, src, src, fsrc, 1, 1);
  // CHECK: call {{.*}} @llvm.nvvm.wmma.m16n16k16.mma.col.row.f32.f32
  // expected-error@+1 {{'__hmma_m16n16k16_mma_f32f32' needs target feature ptx60}}
  __hmma_m16n16k16_mma_f32f32(fdst, src, src, fsrc, 2, 0);
  // CHECK: call {{.*}} @llvm.nvvm.wmma.m16n16k16.mma.col.row.f32.f32.satfinite
  // expected-error@+1 {{'__hmma_m16n16k16_mma_f32f32' needs target feature ptx60}}
  __hmma_m16n16k16_mma_f32f32(fdst, src, src, fsrc, 2, 1);
  // CHECK: call {{.*}} @llvm.nvvm.wmma.m16n16k16.mma.col.col.f32.f32
  // expected-error@+1 {{'__hmma_m16n16k16_mma_f32f32' needs target feature ptx60}}
  __hmma_m16n16k16_mma_f32f32(fdst, src, src, fsrc, 3, 0);
  // CHECK: call {{.*}} @llvm.nvvm.wmma.m16n16k16.mma.col.col.f32.f32.satfinite
  // expected-error@+1 {{'__hmma_m16n16k16_mma_f32f32' needs target feature ptx60}}
  __hmma_m16n16k16_mma_f32f32(fdst, src, src, fsrc, 3, 1);
}
