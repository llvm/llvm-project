// RUN: %clang_cc1 -finclude-default-header -x hlsl -triple dxil-pc-shadermodel6.3-library %s -fnative-half-type -fnative-int16-type -emit-llvm -disable-llvm-passes -o - | FileCheck %s


// CHECK-LABEL: builtin_rcp_half
// CHECK: %hlsl.rcp = fdiv reassoc nnan ninf nsz arcp afn half 0xH3C00, %{{.*}}
// CHECK: ret half  %hlsl.rcp
half builtin_rcp_half(half p0) {
  return __builtin_hlsl_elementwise_rcp(p0);
}

// CHECK-LABEL: builtin_rcp_float
// CHECK: %hlsl.rcp = fdiv reassoc nnan ninf nsz arcp afn float  1.000000e+00, %{{.*}}
// CHECK: ret float  %hlsl.rcp
float builtin_rcp_float(float p0) {
  return __builtin_hlsl_elementwise_rcp(p0);
}
