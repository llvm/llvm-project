// REQUIRES: target=hexagon{{.*}} || target-aarch64 || target-x86_64
// RUN: %clang -S -fenable-ripple -O0 -emit-llvm %s -o - 2>&1 | FileCheck %s --implicit-check-not="warning:" --implicit-check-not="error:"

#include <ripple.h>
#define N 1024
#define VECTOR_LANE 0

#define gen_test(T) \
void blocked_copy_32x4(const T (*base)[2][2][2], T *out) { \
    ripple_block_t BS = ripple_set_block_shape(VECTOR_LANE, 2, 2, 2, 32); \
\
    size_t v0 = ripple_id(BS, 0); \
    size_t v1 = ripple_id(BS, 1); \
    size_t v2 = ripple_id(BS, 2); \
    size_t v3 = ripple_id(BS, 3); \
 \
    T t0 = base[0][v2][v1][v0]; \
    T t1 = base[1][v2][v1][v0]; \
    T t2 = base[2][v2][v1][v0]; \
    T t3 = base[3][v2][v1][v0]; \
    T t4 = base[4][v2][v1][v0]; \
    T t5 = base[5][v2][v1][v0]; \
    T t6 = base[6][v2][v1][v0]; \
    T t7 = base[7][v2][v1][v0]; \
    T t8 = base[8][v2][v1][v0]; \
    T t9 = base[9][v2][v1][v0]; \
    T t10 = base[10][v2][v1][v0]; \
    T t11 = base[11][v2][v1][v0]; \
    T t12 = base[12][v2][v1][v0]; \
    T t13 = base[13][v2][v1][v0]; \
    T t14 = base[14][v2][v1][v0]; \
    T t15 = base[15][v2][v1][v0]; \
    T t16 = base[16][v2][v1][v0]; \
    T t17 = base[17][v2][v1][v0]; \
    T t18 = base[18][v2][v1][v0]; \
    T t19 = base[19][v2][v1][v0]; \
    T t20 = base[20][v2][v1][v0]; \
    T t21 = base[21][v2][v1][v0]; \
    T t22 = base[22][v2][v1][v0]; \
    T t23 = base[23][v2][v1][v0]; \
    T t24 = base[24][v2][v1][v0]; \
    T t25 = base[25][v2][v1][v0]; \
    T t26 = base[26][v2][v1][v0]; \
    T t27 = base[27][v2][v1][v0]; \
    T t28 = base[28][v2][v1][v0]; \
    T t29 = base[29][v2][v1][v0]; \
    T t30 = base[30][v2][v1][v0]; \
    T t31 = base[31][v2][v1][v0]; \
 \
    out[v0 + 2*v1 + 4*v2 + 8 * v3] = \
        ripple_stack(BS, t0, t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11, \
                        t12, t13, t14, t15, t16, t17, t18, t19, t20, t21, \
                        t22, t23, t24, t25, t26, t27, t28, t29, t30, t31); \
}

gen_test(uint8_t)
// CHECK-NOT: ripple.stack
// CHECK-COUNT-32: call <256 x i8> @llvm.vector.insert.v256i8.v8i8
gen_test(int8_t)
// CHECK-NOT: ripple.stack
// CHECK-COUNT-32: call <256 x i8> @llvm.vector.insert.v256i8.v8i8
gen_test(uint16_t)
// CHECK-NOT: ripple.stack
// CHECK-COUNT-32: call <256 x i16> @llvm.vector.insert.v256i16.v8i16
gen_test(int16_t)
// CHECK-NOT: ripple.stack
// CHECK-COUNT-32: call <256 x i16> @llvm.vector.insert.v256i16.v8i16
gen_test(uint32_t)
// CHECK-NOT: ripple.stack
// CHECK-COUNT-32: call <256 x i32> @llvm.vector.insert.v256i32.v8i32
gen_test(int32_t)
// CHECK-NOT: ripple.stack
// CHECK-COUNT-32: call <256 x i32> @llvm.vector.insert.v256i32.v8i32
gen_test(uint64_t)
// CHECK-NOT: ripple.stack
// CHECK-COUNT-32: call <256 x i64> @llvm.vector.insert.v256i64.v8i64
gen_test(int64_t)
// CHECK-NOT: ripple.stack
// CHECK-COUNT-32: call <256 x i64> @llvm.vector.insert.v256i64.v8i64

gen_test(float)
// CHECK-NOT: ripple.stack
// CHECK-COUNT-32: call <256 x float> @llvm.vector.insert.v256f32.v8f32
gen_test(double)
// CHECK-NOT: ripple.stack
// CHECK-COUNT-32: call <256 x double> @llvm.vector.insert.v256f64.v8f64