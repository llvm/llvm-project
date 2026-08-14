// RUN: %clang_cc1 -finclude-default-header -x hlsl -triple dxil-pc-shadermodel6.3-library %s -fnative-half-type -fnative-int16-type -emit-llvm -O1 -o - | FileCheck %s


// CHECK: define {{.*}}test_vector_uint{{.*}}(<16 x i32> {{.*}} [[VAL:%.*]]){{.*}} 
// CHECK: bitcast <16 x i32> [[VAL]] to <16 x float>
float4x4 test_vector_uint(uint4x4 p0) {
  return asfloat(p0);
}

// CHECK: define {{.*}}test_vector_int{{.*}}(<16 x i32> {{.*}} [[VAL:%.*]]){{.*}} 
// CHECK: bitcast <16 x i32> [[VAL]] to <16 x float>
float4x4 test_vector_int(int4x4 p0) {
  return asfloat(p0);
}

// CHECK: define {{.*}}test_vector_float{{.*}}(<16 x float> {{.*}} [[VAL:%.*]]){{.*}} 
// CHECK-NOT: bitcast
// CHECK: ret <16 x float> [[VAL]]
float4x4 test_vector_float(float4x4 p0) {
  return asfloat(p0);
}
