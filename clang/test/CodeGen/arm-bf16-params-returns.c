// RUN: %clang_cc1 -triple armv8.6a-arm-none-eabi -target-abi aapcs -mfloat-abi hard -target-feature +bf16 -target-feature +neon -emit-llvm -O2 -o - %s | opt -S -passes=mem2reg,sroa | FileCheck %s --check-prefixes=CHECK32-HARD,CHECK
// RUN: %clang_cc1 -triple armv8.6a-arm-none-eabi -target-abi aapcs -mfloat-abi softfp -target-feature +bf16 -target-feature +neon -emit-llvm -O2 -o - %s | opt -S -passes=mem2reg,sroa | FileCheck %s --check-prefixes=CHECK32-SOFTFP,CHECK
// RUN: %clang_cc1 -triple aarch64 -target-abi aapcs -target-feature +bf16 -target-feature +neon -emit-llvm -O2 -o - %s | opt -S -passes=mem2reg,sroa | FileCheck %s --check-prefixes=CHECK64,CHECK64NEON,CHECK
// RUN: %clang_cc1 -triple aarch64 -target-abi aapcs -target-feature -bf16 -target-feature +neon -DNONEON -emit-llvm -O2 -o - %s | opt -S -passes=mem2reg,sroa | FileCheck %s --check-prefixes=CHECK64,CHECK

// REQUIRES: aarch64-registered-target || arm-registered-target

#include <arm_neon.h>

// function return types
__bf16 test_ret_bf16(__bf16 v) {
  return v;
}
// CHECK32-HARD: define{{.*}} arm_aapcs_vfpcc noundef bfloat @test_ret_bf16(bfloat noundef returned %v) {{.*}} {
// CHECK32-HARD: ret bfloat %v
// CHECK32-SOFTFP: define{{.*}} bfloat @test_ret_bf16(bfloat noundef returned %v) {{.*}} {
// CHECK32-SOFTFP: ret bfloat %v
// CHECK64: define{{.*}} bfloat @test_ret_bf16(bfloat noundef returned %v) {{.*}} {
// CHECK64: ret bfloat %v

// GH170434
__fp16 test_ret_fp16_from_bf16(__bf16 v) {
  return v;
}
// CHECK: define{{.*}} half @test_ret_fp16_from_bf16(bfloat noundef %v) {{.*}} {
// CHECK: [[EXT:%.*]] = fpext bfloat %v to float
// CHECK: [[TRUNC:%.*]] = fptrunc float [[EXT]] to half
// CHECK: ret half [[TRUNC]]

__bf16 test_ret_bf16_from_fp16(__fp16 v) {
  return v;
}
// CHECK: define{{.*}} bfloat @test_ret_bf16_from_fp16(half noundef %v) {{.*}} {
// CHECK: [[EXT:%.*]] = fpext half %v to float
// CHECK: [[TRUNC:%.*]] = fptrunc float [[EXT]] to bfloat
// CHECK: ret bfloat [[TRUNC]]

__bf16 test_fmaxf16_bf16(__bf16 a, __bf16 b) {
  return __builtin_fmaxf16(a, b);
}
// CHECK: define{{.*}} bfloat @test_fmaxf16_bf16(bfloat noundef %a, bfloat noundef %b) {{.*}} {
// CHECK: [[EXTA:%.*]] = fpext nsz bfloat %a to float
// CHECK: [[TRUNCA:%.*]] = fptrunc nsz float [[EXTA]] to half
// CHECK: [[EXTB:%.*]] = fpext nsz bfloat %b to float
// CHECK: [[TRUNCB:%.*]] = fptrunc nsz float [[EXTB]] to half
// CHECK: [[MAX:%.*]] = {{.*}}call nsz half @llvm.maxnum.f16(half [[TRUNCA]], half [[TRUNCB]])
// CHECK: [[EXT:%.*]] = fpext half [[MAX]] to float
// CHECK: [[TRUNC:%.*]] = fptrunc float [[EXT]] to bfloat
// CHECK: ret bfloat [[TRUNC]]

#ifndef NONEON

bfloat16x4_t test_ret_bf16x4_t(bfloat16x4_t v) {
  return v;
}
// CHECK32-HARD: define{{.*}} arm_aapcs_vfpcc noundef <4 x bfloat> @test_ret_bf16x4_t(<4 x bfloat> noundef returned %v) {{.*}} {
// CHECK32-HARD: ret <4 x bfloat> %v
// CHECK32-SOFTFP: define{{.*}} noundef <2 x i32> @test_ret_bf16x4_t(<2 x i32> [[V0:.*]]) {{.*}} {
// CHECK32-SOFTFP: ret <2 x i32> %v
// CHECK64NEON: define{{.*}} noundef <4 x bfloat> @test_ret_bf16x4_t(<4 x bfloat> noundef returned %v) {{.*}} {
// CHECK64NEON: ret <4 x bfloat> %v

#endif