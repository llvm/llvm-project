// RUN: %clang_cc1 -std=hlsl2021 -finclude-default-header -triple \
// RUN:   dxil-pc-shadermodel6.3-compute %s -emit-llvm \
// RUN:   -disable-llvm-passes -o - |  FileCheck %s --check-prefixes=CHECK,CHECK-DXIL

// RUN: %clang_cc1 -std=hlsl2021 -finclude-default-header -triple \
// RUN:   spirv-pc-vulkan-compute %s -emit-llvm  \
// RUN:   -disable-llvm-passes -o - | FileCheck %s --check-prefixes=CHECK,CHECK-SPIRV

// Test basic lowering to runtime function call.

int test_int(bool expr) {
// CHECK-DXIL: define hidden noundef i32 {{.*}}(i1 noundef %[[EXPR:.*]]) #[[CONVATTR:.*]] {
// CHECK-SPIRV: define hidden spir_func noundef i32 {{.*}}(i1 noundef %[[EXPR:.*]]) #[[CONVATTR:.*]] {
  // CHECK: entry:
  // CHECK: %[[EXPRADDR:.*]] = alloca i32, align 4
  // CHECK: %[[STOREDVAL:.*]] = zext i1 %[[EXPR]] to i32
  // CHECK: store i32 %[[STOREDVAL]], ptr %[[EXPRADDR]], align 4
  // CHECK: %[[LOADEDVAL:.*]] = load i32, ptr %[[EXPRADDR]], align 4
  // CHECK: %[[TRUNCLOADEDVAL:.*]] = trunc i32 %[[LOADEDVAL]] to i1

  // CHECK-SPIRV:  %[[RET:.*]] = call spir_func [[TY:.*]] @llvm.spv.subgroup.prefix.bit.count(i1 %[[TRUNCLOADEDVAL]])
  // CHECK-DXIL:  %[[RET:.*]] = call [[TY:.*]] @llvm.dx.wave.prefix.bit.count(i1 %[[TRUNCLOADEDVAL]])
  // CHECK: ret [[TY]] %[[RET]]
  return WavePrefixCountBits(expr);
}

// CHECK: attributes #[[CONVATTR]] = {{{.*}} convergent {{.*}}}
