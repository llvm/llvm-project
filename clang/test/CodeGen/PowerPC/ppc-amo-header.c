// REQUIRES: powerpc-registered-target
// RUN: %clang_cc1 -triple powerpc64le-unknown-linux-gnu -target-cpu pwr9 \
// RUN:   -emit-llvm %s -o - | FileCheck %s
// RUN: %clang_cc1 -triple powerpc64-ibm-aix -target-cpu pwr9 \
// RUN:   -emit-llvm %s -o - | FileCheck %s

#include <amo.h>

uint32_t test_lwat_add(uint32_t *ptr, uint32_t val) {
  // CHECK-LABEL: @test_lwat_add
  // CHECK: call i32 @llvm.ppc.amo.lwat(ptr %{{.*}}, i32 %{{.*}}, i32 0)
  return amo_lwat_add(ptr, val);
}

uint32_t test_lwat_xor(uint32_t *ptr, uint32_t val) {
  // CHECK-LABEL: @test_lwat_xor
  // CHECK: call i32 @llvm.ppc.amo.lwat(ptr %{{.*}}, i32 %{{.*}}, i32 1)
  return amo_lwat_xor(ptr, val);
}

uint32_t test_lwat_ior(uint32_t *ptr, uint32_t val) {
  // CHECK-LABEL: @test_lwat_ior
  // CHECK: call i32 @llvm.ppc.amo.lwat(ptr %{{.*}}, i32 %{{.*}}, i32 2)
  return amo_lwat_ior(ptr, val);
}

uint32_t test_lwat_and(uint32_t *ptr, uint32_t val) {
  // CHECK-LABEL: @test_lwat_and
  // CHECK: call i32 @llvm.ppc.amo.lwat(ptr %{{.*}}, i32 %{{.*}}, i32 3)
  return amo_lwat_and(ptr, val);
}

uint32_t test_lwat_umax(uint32_t *ptr, uint32_t val) {
  // CHECK-LABEL: @test_lwat_umax
  // CHECK: call i32 @llvm.ppc.amo.lwat(ptr %{{.*}}, i32 %{{.*}}, i32 4)
  return amo_lwat_umax(ptr, val);
}

uint32_t test_lwat_umin(uint32_t *ptr, uint32_t val) {
  // CHECK-LABEL: @test_lwat_umin
  // CHECK: call i32 @llvm.ppc.amo.lwat(ptr %{{.*}}, i32 %{{.*}}, i32 6)
  return amo_lwat_umin(ptr, val);
}

uint32_t test_lwat_swap(uint32_t *ptr, uint32_t val) {
  // CHECK-LABEL: @test_lwat_swap
  // CHECK: call i32 @llvm.ppc.amo.lwat(ptr %{{.*}}, i32 %{{.*}}, i32 8)
  return amo_lwat_swap(ptr, val);
}

int32_t test_lwat_sadd(int32_t *ptr, int32_t val) {
  // CHECK-LABEL: @test_lwat_sadd
  // CHECK: call i32 @llvm.ppc.amo.lwat(ptr %{{.*}}, i32 %{{.*}}, i32 0)
  return amo_lwat_sadd(ptr, val);
}

int32_t test_lwat_smax(int32_t *ptr, int32_t val) {
  // CHECK-LABEL: @test_lwat_smax
  // CHECK: call i32 @llvm.ppc.amo.lwat(ptr %{{.*}}, i32 %{{.*}}, i32 5)
  return amo_lwat_smax(ptr, val);
}

int32_t test_lwat_smin(int32_t *ptr, int32_t val) {
  // CHECK-LABEL: @test_lwat_smin
  // CHECK: call i32 @llvm.ppc.amo.lwat(ptr %{{.*}}, i32 %{{.*}}, i32 7)
  return amo_lwat_smin(ptr, val);
}

int32_t test_lwat_sswap(int32_t *ptr, int32_t val) {
  // CHECK-LABEL: @test_lwat_sswap
  // CHECK: call i32 @llvm.ppc.amo.lwat(ptr %{{.*}}, i32 %{{.*}}, i32 8)
  return amo_lwat_sswap(ptr, val);
}

uint64_t test_ldat_add(uint64_t *ptr, uint64_t val) {
  // CHECK-LABEL: @test_ldat_add
  // CHECK: call i64 @llvm.ppc.amo.ldat(ptr %{{.*}}, i64 %{{.*}}, i32 0)
  return amo_ldat_add(ptr, val);
}

uint64_t test_ldat_xor(uint64_t *ptr, uint64_t val) {
  // CHECK-LABEL: @test_ldat_xor
  // CHECK: call i64 @llvm.ppc.amo.ldat(ptr %{{.*}}, i64 %{{.*}}, i32 1)
  return amo_ldat_xor(ptr, val);
}

uint64_t test_ldat_ior(uint64_t *ptr, uint64_t val) {
  // CHECK-LABEL: @test_ldat_ior
  // CHECK: call i64 @llvm.ppc.amo.ldat(ptr %{{.*}}, i64 %{{.*}}, i32 2)
  return amo_ldat_ior(ptr, val);
}

uint64_t test_ldat_and(uint64_t *ptr, uint64_t val) {
  // CHECK-LABEL: @test_ldat_and
  // CHECK: call i64 @llvm.ppc.amo.ldat(ptr %{{.*}}, i64 %{{.*}}, i32 3)
  return amo_ldat_and(ptr, val);
}

uint64_t test_ldat_umax(uint64_t *ptr, uint64_t val) {
  // CHECK-LABEL: @test_ldat_umax
  // CHECK: call i64 @llvm.ppc.amo.ldat(ptr %{{.*}}, i64 %{{.*}}, i32 4)
  return amo_ldat_umax(ptr, val);
}

uint64_t test_ldat_umin(uint64_t *ptr, uint64_t val) {
  // CHECK-LABEL: @test_ldat_umin
  // CHECK: call i64 @llvm.ppc.amo.ldat(ptr %{{.*}}, i64 %{{.*}}, i32 6)
  return amo_ldat_umin(ptr, val);
}

uint64_t test_ldat_swap(uint64_t *ptr, uint64_t val) {
  // CHECK-LABEL: @test_ldat_swap
  // CHECK: call i64 @llvm.ppc.amo.ldat(ptr %{{.*}}, i64 %{{.*}}, i32 8)
  return amo_ldat_swap(ptr, val);
}

int64_t test_ldat_sadd(int64_t *ptr, int64_t val) {
  // CHECK-LABEL: @test_ldat_sadd
  // CHECK: call i64 @llvm.ppc.amo.ldat(ptr %{{.*}}, i64 %{{.*}}, i32 0)
  return amo_ldat_sadd(ptr, val);
}

int64_t test_ldat_smax(int64_t *ptr, int64_t val) {
  // CHECK-LABEL: @test_ldat_smax
  // CHECK: call i64 @llvm.ppc.amo.ldat(ptr %{{.*}}, i64 %{{.*}}, i32 5)
  return amo_ldat_smax(ptr, val);
}

int64_t test_ldat_smin(int64_t *ptr, int64_t val) {
  // CHECK-LABEL: @test_ldat_smin
  // CHECK: call i64 @llvm.ppc.amo.ldat(ptr %{{.*}}, i64 %{{.*}}, i32 7)
  return amo_ldat_smin(ptr, val);
}

int64_t test_ldat_sswap(int64_t *ptr, int64_t val) {
  // CHECK-LABEL: @test_ldat_sswap
  // CHECK: call i64 @llvm.ppc.amo.ldat(ptr %{{.*}}, i64 %{{.*}}, i32 8)
  return amo_ldat_sswap(ptr, val);
}
