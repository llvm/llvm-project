// RUN: %clang_cc1 -finclude-default-header -x hlsl -triple dxil-pc-shadermodel6.2-library %s -fnative-half-type -emit-llvm -O1 -o - | FileCheck %s
// CHECK: define {{.*}}test_ints{{.*}}(i16 {{.*}} [[VAL:%.*]]){{.*}}
// CHECK-NOT: bitcast
// CHECK: ret i16 [[VAL]]
int16_t test_int(int16_t p0)
{
    return asint16(p0);
}

//CHECK: define {{.*}}test_uint{{.*}}(i16 {{.*}} [[VAL:%.*]]){{.*}}
//CHECK-NOT: bitcast
//CHECK: ret i16 [[VAL]]
int16_t test_uint(uint16_t p0)
{
    return asint16(p0);
}

//CHECK: define {{.*}}test_half{{.*}}(half {{.*}} [[VAL:%.*]]){{.*}}
//CHECK: [[RES:%.*]] = bitcast half [[VAL]] to i16
//CHECK : ret i16 [[RES]]
int16_t test_half(half p0)
{
    return asint16(p0);
}

//CHECK: define {{.*}}test_vector_int{{.*}}(<4 x i16> {{.*}} [[VAL:%.*]]){{.*}}
//CHECK-NOT: bitcast
//CHECK: ret <4 x i16> [[VAL]]
int16_t4 test_vector_int(int16_t4 p0)
{
    return asint16(p0);
}

//CHECK: define {{.*}}test_vector_uint{{.*}}(<4 x i16> {{.*}} [[VAL:%.*]]){{.*}}
//CHECK-NOT: bitcast
//CHECK: ret <4 x i16> [[VAL]]
int16_t4 test_vector_uint(uint16_t4 p0)
{
    return asint16(p0);
}

//CHECK: define {{.*}}fn{{.*}}(<4 x half> {{.*}} [[VAL:%.*]]){{.*}}
//CHECK: [[RES:%.*]] = bitcast <4 x half> [[VAL]] to <4 x i16>
//CHECK: ret <4 x i16> [[RES]]
int16_t4 fn(half4 p1)
{
    return asint16(p1);
}