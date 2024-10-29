// RUN: %clang_cc1 -finclude-default-header -x hlsl -triple dxil-pc-shadermodel6.3-pixel %s -fnative-half-type -emit-llvm -o - | FileCheck %s

RWBuffer<float4> Buf;

//CHECK-LABEL: define void @main()
float4 main( ) {
  float4 p1 = Buf[0];
  //CHECK: [[LOAD:%.*]] = load <4 x float>, ptr %p1{{.*}}, align 16
  //CHECK-NEXT: [[EXTR:%.*]] = extractelement <4 x float> [[LOAD]], i32 3
  //CHECK-NEXT: [[FCMP:%.*]] = fcmp olt float [[EXTR]], 0.000000e+00
  //CHECK-NEXT: call void @llvm.dx.clip(i1 [[FCMP]])
  clip(p1.a);
  return p1;
}
