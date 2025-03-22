// RUN: %clang_cc1 -finclude-default-header -fnative-half-type -triple \
// RUN:   dxil-pc-shadermodel6.3-compute %s -emit-llvm -disable-llvm-passes -o - | \
// RUN:   FileCheck %s --check-prefixes=CHECK,CHECK-DXIL
// RUN: %clang_cc1 -finclude-default-header -fnative-half-type -triple \
// RUN:   spirv-pc-vulkan-compute %s -emit-llvm -disable-llvm-passes -o - | \
// RUN:   FileCheck %s --check-prefixes=CHECK,CHECK-SPIRV

// Test basic lowering to runtime function call.

float test(half2 p1, half2 p2, float p3) {
  // CHECK-SPIRV:  %[[MUL:.*]] = call {{.*}} half @llvm.spv.fdot.v2f16(<2 x half> %1, <2 x half> %2)
  // CHECK-SPIRV:  %[[CONVERT:.*]] = fpext {{.*}} half %[[MUL:.*]] to float
  // CHECK-SPIRV:  %[[C:.*]] = load float, ptr %c.addr, align 4
  // CHECK-SPIRV:  %[[RES:.*]] = fadd {{.*}} float %[[CONVERT:.*]], %[[C:.*]]
  // CHECK-DXIL:  %[[RES:.*]] = call {{.*}} float @llvm.dx.dot2add.v2f16(<2 x half> %0, <2 x half> %1, float %2)
  // CHECK:  ret float %[[RES]]
  return dot2add(p1, p2, p3);
}
