// RUN: %clang_cc1 -std=hlsl2021 -finclude-default-header -x hlsl -triple \
// RUN:   dxil-pc-shadermodel6.3-library %s -fnative-half-type \
// RUN:   -emit-llvm -disable-llvm-passes -o - | FileCheck %s

// Make sure float3 is not changed into float4.
// CHECK:<3 x float> @_Z3fooDv3_f(<3 x float> noundef nofpclass(nan inf) %[[PARAM:[0-9a-zA-Z]+]])
// CHECK:%[[A_ADDR:.+]] = alloca <3 x float>, align 16
// CHECK-NEXT:store <3 x float> %[[PARAM]], ptr %[[A_ADDR]], align 16
// CHECK-NEXT:%[[V:[0-9]+]] = load <3 x float>, ptr %[[A_ADDR]], align 16
// CHECK-NEXT:ret <3 x float> %[[V]]
float3 foo(float3 a) {
  return a;
}
