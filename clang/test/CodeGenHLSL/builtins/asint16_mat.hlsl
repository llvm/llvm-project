// RUN: %clang_cc1 -finclude-default-header -x hlsl -triple dxil-pc-shadermodel6.3-library %s -fnative-half-type -fnative-int16-type -emit-llvm -O1 -o - | FileCheck %s

// CHECK: define {{.*}}test_matrix_int{{.*}}(<16 x i16> {{.*}} [[VAL:%.*]]){{.*}}
// CHECK-NOT: bitcast
// CHECK: ret <16 x i16> [[VAL]]
int16_t4x4 test_matrix_int(int16_t4x4 p0) {
  return asint16(p0);
}

// CHECK: define {{.*}}test_matrix_uint{{.*}}(<16 x i16> {{.*}} [[VAL:%.*]]){{.*}}
// CHECK-NOT: bitcast
// CHECK: ret <16 x i16> [[VAL]]
int16_t4x4 test_matrix_uint(uint16_t4x4 p0) {
  return asint16(p0);
}

// CHECK: define {{.*}}test_matrix_half{{.*}}(<16 x half> {{.*}} [[VAL:%.*]]){{.*}}
// CHECK: bitcast <16 x half> [[VAL]] to <16 x i16>
int16_t4x4 test_matrix_half(half4x4 p0) {
  return asint16(p0);
}
