// RUN: %clang_cc1 -finclude-default-header -x hlsl -triple dxil-pc-shadermodel6.3-library %s -fnative-half-type -emit-llvm -O1 -o - | FileCheck %s

// CHECK: define {{.*}}test_uint{{.*}}(i32 {{.*}} [[VAL:%.*]]){{.*}} 
// CHECK: bitcast i32 [[VAL]] to float
float test_uint(uint p0) {
  return asfloat(p0);
}

// CHECK: define {{.*}}test_int{{.*}}(i32 {{.*}} [[VAL:%.*]]){{.*}} 
// CHECK: bitcast i32 [[VAL]] to float
float test_int(int p0) {
  return asfloat(p0);
}

// CHECK: define {{.*}}test_float{{.*}}(float {{.*}} [[VAL:%.*]]){{.*}} 
// CHECK-NOT: bitcast
// CHECK: ret float [[VAL]]
float test_float(float p0) {
  return asfloat(p0);
}

// CHECK: define {{.*}}test_vector_uint{{.*}}(<4 x i32> {{.*}} [[VAL:%.*]]){{.*}} 
// CHECK: bitcast <4 x i32> [[VAL]] to <4 x float>

float4 test_vector_uint(uint4 p0) {
  return asfloat(p0);
}

// CHECK: define {{.*}}test_vector_int{{.*}}(<4 x i32> {{.*}} [[VAL:%.*]]){{.*}} 
// CHECK: bitcast <4 x i32> [[VAL]] to <4 x float>
float4 test_vector_int(int4 p0) {
  return asfloat(p0);
}

// CHECK: define {{.*}}test_vector_float{{.*}}(<4 x float> {{.*}} [[VAL:%.*]]){{.*}} 
// CHECK-NOT: bitcast
// CHECK: ret <4 x float> [[VAL]]
float4 test_vector_float(float4 p0) {
  return asfloat(p0);
}
