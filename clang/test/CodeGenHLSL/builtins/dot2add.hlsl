// RUN: %clang_cc1 -finclude-default-header -triple \
// RUN:   dxil-pc-shadermodel6.3-compute %s -emit-llvm -disable-llvm-passes -o - | \
// RUN:   FileCheck %s --check-prefixes=CHECK,CHECK-DXIL
// RUN: %clang_cc1 -finclude-default-header -triple \
// RUN:   spirv-pc-vulkan-compute %s -emit-llvm -disable-llvm-passes -o - | \
// RUN:   FileCheck %s --check-prefixes=CHECK,CHECK-SPIRV

// Test basic lowering to runtime function call.

float test(half2 p1, half2 p2, float p3) {
  // CHECK-SPIRV:  %[[MUL:.*]] = call {{.*}} float @llvm.spv.fdot.v2f32(<2 x float> %1, <2 x float> %2)
  // CHECK-SPIRV:  %[[C:.*]] = load float, ptr %c.addr, align 4
  // CHECK-SPIRV:  %[[RES:.*]] = fadd {{.*}} float %[[MUL]], %[[C]]
  // CHECK-DXIL:  %[[RES:.*]] = call {{.*}} float @llvm.dx.dot2add.v2f32(<2 x float> %0, <2 x float> %1, float %2)
  // CHECK:  ret float %[[RES]]
  return dot2add(p1, p2, p3);
}