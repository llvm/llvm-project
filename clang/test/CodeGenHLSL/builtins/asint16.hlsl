// RUN: %clang_cc1 -finclude-default-header -x hlsl -triple dxil-pc-shadermodel6.2-library %s -fnative-half-type -emit-llvm -O1 -o - | FileCheck %s

//CHECK-LABEL: define {{.*}}test_ints
//CHECK-SAME: {{.*}}(i16 {{.*}} [[VAL:%.*]]){{.*}}
//CHECK-NOT: bitcast
//CHECK: entry:
//CHECK-NEXT: ret i16 [[VAL]]
int16_t test_int(int16_t p0)
{
    return asint16(p0);
}

//CHECK-LABEL: define {{.*}}test_uint
//CHECK-SAME: {{.*}}(i16 {{.*}} [[VAL:%.*]]){{.*}}
//CHECK-NOT:bitcast
//CHECK: entry:
//CHECK-NEXT: ret i16 [[VAL]]
int16_t test_uint(uint16_t p0)
{
    return asint16(p0);
}

//CHECK-LABEL: define {{.*}}test_half
//CHECK-SAME: {{.*}}(half {{.*}} [[VAL:%.*]]){{.*}}
//CHECK: [[RES:%.*]] = bitcast half [[VAL]] to i16
//CHECK-NEXT : ret i16 [[RES]]
int16_t test_half(half p0)
{
    return asint16(p0);
}

//CHECK-LABEL: define {{.*}}test_vector_int
//CHECK-SAME: {{.*}}(<4 x i16> {{.*}} [[VAL:%.*]]){{.*}}
//CHECK-NOT: bitcast
//CHECK: entry:
//CHECK-NEXT: ret <4 x i16> [[VAL]]
int16_t4 test_vector_int(int16_t4 p0)
{
    return asint16(p0);
}

//CHECK-LABEL: define {{.*}}test_vector_uint
//CHECK-SAME: {{.*}}(<4 x i16> {{.*}} [[VAL:%.*]]){{.*}}
//CHECK-NOT: bitcast
//CHECK-NEXT: entry:
//CHECK-NEXT: ret <4 x i16> [[VAL]]
int16_t4 test_vector_uint(uint16_t4 p0)
{
    return asint16(p0);
}

//CHECK-LABEL: define {{.*}}fn
//CHECK-SAME: {{.*}}(<4 x half> {{.*}} [[VAL:%.*]]){{.*}}
//CHECK: [[RES:%.*]] = bitcast <4 x half> [[VAL]] to <4 x i16>
//CHECK-NEXT: ret <4 x i16> [[RES]]
int16_t4 fn(half4 p1)
{
    return asint16(p1);
}

