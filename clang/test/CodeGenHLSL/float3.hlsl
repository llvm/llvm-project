// RUN: %clang --driver-mode=dxc -Tlib_6_7 -fcgl  -Fo - %s | FileCheck %s

// Make sure float3 is not changed into float4.
// CHECK:<3 x float> @"?foo@@YAT?$__vector@M$02@__clang@@T12@@Z"(<3 x float> noundef %a)
// CHECK:%[[A_ADDR:.+]] = alloca <3 x float>, align 16
// CHECK-NEXT:store <3 x float> %a, ptr %[[A_ADDR]], align 16
// CHECK-NEXT:%[[V:[0-9]+]] = load <3 x float>, ptr %[[A_ADDR]], align 16
// CHECK-NEXT:ret <3 x float> %[[V]]
float3 foo(float3 a) {
  return a;
}
