// RUN: %clang_cc1 -finclude-default-header -triple \
// RUN:   dxil-pc-shadermodel6.3-compute %s -emit-llvm -disable-llvm-passes -o - | \
// RUN:   FileCheck %s -DTARGET=dx

// Test basic lowering to runtime function call.

// CHECK-LABEL: test
float test(half2 p1, half2 p2, float p3) {
  // CHECK-DXIL:  %hlsl.dot2add = call reassoc nnan ninf nsz arcp afn float @llvm.dx.dot2add.v2f32(<2 x float> %0, <2 x float> %1, float %2)
  // CHECK: ret float %hlsl.dot2add
  return dot2add(p1, p2, p3);
}

// CHECK: declare [[TY]] @llvm.[[TARGET]].dot4add.i8packed([[TY]], [[TY]], [[TY]])
