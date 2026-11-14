// RUN: %clang_cc1 -finclude-default-header -triple dxil-pc-shadermodel6.6-library %s -emit-llvm -disable-llvm-passes -o - | FileCheck %s

using namespace hlsl;

// CHECK-LABEL: test_mul_square
// CHECK: fmul {{.*}}<4 x float> %{{.*}}, %{{.*}}
float2x2 test_mul_square(float2x2 a, float2x2 b) {
  return a * b;
}

// CHECK-LABEL: test_mul_float
// CHECK: fmul {{.*}}<6 x float> %{{.*}}, %{{.*}}
float2x3 test_mul_float(float2x3 a, float2x3 b) {
  return a * b;
}

// CHECK-LABEL: test_div_float
// CHECK: fdiv {{.*}}<6 x float> %{{.*}}, %{{.*}}
float2x3 test_div_float(float2x3 a, float2x3 b) {
  return a / b;
}

// CHECK-LABEL: test_neg_float
// CHECK: fneg {{.*}}<6 x float> %{{.*}}
float2x3 test_neg_float(float2x3 a) {
  return -a;
}

// CHECK-LABEL: test_mul_int
// CHECK: mul <6 x i32> %{{.*}}, %{{.*}}
int2x3 test_mul_int(int2x3 a, int2x3 b) {
  return a * b;
}

// CHECK-LABEL: test_div_int
// CHECK: sdiv <6 x i32> %{{.*}}, %{{.*}}
int2x3 test_div_int(int2x3 a, int2x3 b) {
  return a / b;
}

// CHECK-LABEL: test_div_uint
// CHECK: udiv <6 x i32> %{{.*}}, %{{.*}}
uint2x3 test_div_uint(uint2x3 a, uint2x3 b) {
  return a / b;
}

// CHECK-LABEL: test_neg_int
// CHECK: sub <6 x i32> zeroinitializer, %{{.*}}
int2x3 test_neg_int(int2x3 a) {
  return -a;
}
