// RUN: %clang_cc1 -triple armv8.6a-arm-none-eabi -target-abi aapcs -mfloat-abi hard -target-feature +bf16 -target-feature +neon -emit-llvm -O2 -o - %s | opt -S -passes=mem2reg,sroa | FileCheck %s --check-prefix=CHECK32-HARD
// RUN: %clang_cc1 -triple armv8.6a-arm-none-eabi -target-abi aapcs -mfloat-abi softfp -target-feature +bf16 -target-feature +neon -emit-llvm -O2 -o - %s | opt -S -passes=mem2reg,sroa | FileCheck %s --check-prefix=CHECK32-SOFTFP
// RUN: %clang_cc1 -triple aarch64 -target-abi aapcs -target-feature +bf16 -target-feature +neon -emit-llvm -O2 -o - %s | opt -S -passes=mem2reg,sroa | FileCheck %s --check-prefixes=CHECK64,CHECK64NEON
// RUN: %clang_cc1 -triple aarch64 -target-abi aapcs -target-feature -bf16 -target-feature +neon -DNONEON -emit-llvm -O2 -o - %s | opt -S -passes=mem2reg,sroa | FileCheck %s --check-prefix=CHECK64

// REQUIRES: aarch64-registered-target || arm-registered-target

#include <arm_neon.h>

// function return types
__bf16 test_ret_bf16(__bf16 v) {
  return v;
}
// CHECK32-HARD: define{{.*}} arm_aapcs_vfpcc bfloat @test_ret_bf16(bfloat noundef returned %v) {{.*}} {
// CHECK32-HARD: ret bfloat %v
// CHECK32-SOFTFP: define{{.*}} bfloat @test_ret_bf16(bfloat noundef returned %v) {{.*}} {
// CHECK32-SOFTFP: ret bfloat %v
// CHECK64: define{{.*}} bfloat @test_ret_bf16(bfloat noundef returned %v) {{.*}} {
// CHECK64: ret bfloat %v

#ifndef NONEON

bfloat16x4_t test_ret_bf16x4_t(bfloat16x4_t v) {
  return v;
}
// CHECK32-HARD: define{{.*}} arm_aapcs_vfpcc <4 x bfloat> @test_ret_bf16x4_t(<4 x bfloat> noundef returned %v) {{.*}} {
// CHECK32-HARD: ret <4 x bfloat> %v
// CHECK32-SOFTFP: define{{.*}} <2 x i32> @test_ret_bf16x4_t(<2 x i32> [[V0:.*]]) {{.*}} {
// CHECK32-SOFTFP: ret <2 x i32> %v
// CHECK64NEON: define{{.*}} <4 x bfloat> @test_ret_bf16x4_t(<4 x bfloat> noundef returned %v) {{.*}} {
// CHECK64NEON: ret <4 x bfloat> %v

#endif