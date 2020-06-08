// RUN: %clang_cc1 -triple armv8.6a-arm-none-eabi -target-abi aapcs -mfloat-abi hard -target-feature +bf16 -target-feature +neon -emit-llvm -O2 -o - %s | opt -S -mem2reg -sroa | FileCheck %s --check-prefix=CHECK32-HARD
// RUN: %clang_cc1 -triple aarch64-arm-none-eabi -target-abi aapcs -mfloat-abi hard -target-feature +bf16 -target-feature +neon -emit-llvm -O2 -o - %s | opt -S -mem2reg -sroa | FileCheck %s --check-prefix=CHECK64-HARD
// RUN: %clang_cc1 -triple armv8.6a-arm-none-eabi -target-abi aapcs -mfloat-abi softfp -target-feature +bf16 -target-feature +neon -emit-llvm -O2 -o - %s | opt -S -mem2reg -sroa | FileCheck %s --check-prefix=CHECK32-SOFTFP
// RUN: %clang_cc1 -triple aarch64-arm-none-eabi -target-abi aapcs -mfloat-abi softfp -target-feature +bf16 -target-feature +neon -emit-llvm -O2 -o - %s | opt -S -mem2reg -sroa | FileCheck %s --check-prefix=CHECK64-SOFTFP

// function return types
__bf16 test_ret_bf16(__bf16 v) {
  return v;
}
// CHECK32-HARD: define arm_aapcs_vfpcc bfloat @test_ret_bf16(bfloat returned %v) {{.*}} {
// CHECK32-HARD: ret bfloat %v
// CHECK64-HARD: define bfloat @test_ret_bf16(bfloat returned %v) {{.*}} {
// CHECK64-HARD: ret bfloat %v
// CHECK32-SOFTFP: define i32 @test_ret_bf16(i32 [[V0:.*]]) {{.*}} {
// CHECK32-SOFTFP: %tmp2.0.insert.ext = and i32 [[V0]], 65535
// CHECK32-SOFTFP: ret i32 %tmp2.0.insert.ext
// CHECK64-SOFTFP: define bfloat @test_ret_bf16(bfloat returned %v) {{.*}} {
// CHECK64-SOFTFP: ret bfloat %v
