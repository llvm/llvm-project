// RUN: %clang_cc1 -finclude-default-header -x hlsl -triple dxil-pc-shadermodel6.3-library %s -fnative-half-type -emit-llvm -disable-llvm-passes -o - | FileCheck %s


// CHECK-LABEL: test_asuint4_uint
// CHECK: ret i32 %0
export uint test_asuint4_uint(uint p0) {
  return asuint(p0);
}

// CHECK-LABEL: test_asuint4_int
// CHECK: %splat.splatinsert = insertelement <4 x i32> poison, i32 %0, i64 0
export uint4 test_asuint4_int(int p0) {
  return asuint(p0);
}

// CHECK-LABEL: test_asuint_float
// CHECK: %1 = bitcast float %0 to i32
export uint test_asuint_float(float p0) {
  return asuint(p0);
}

// CHECK-LABEL: test_asuint_float
// CHECK: %1 = bitcast <4 x float> %0 to <4 x i32>
export uint4 test_asuint_float4(float4 p0) {
  return asuint(p0);
}