// REQUIRES: arm-registered-target
// RUN: %clang_cc1 -triple thumbv7-apple-darwin9 \
// RUN:   -target-abi aapcs \
// RUN:   -target-cpu cortex-a8 \
// RUN:   -mfloat-abi hard \
// RUN:   -ffreestanding \
// RUN:   -emit-llvm -w -o - %s | FileCheck %s

// RUN: %clang_cc1 -triple armv7-unknown-nacl-gnueabi \
// RUN:  -target-cpu cortex-a8 \
// RUN:  -mfloat-abi hard \
// RUN:  -ffreestanding \
// RUN:  -emit-llvm -w -o - %s | FileCheck %s

#include <arm_neon.h>

struct homogeneous_struct {
  float f[2];
  float f3;
  float f4;
};
// CHECK: define arm_aapcs_vfpcc %struct.homogeneous_struct @test_struct(float %{{.*}}, float %{{.*}}, float %{{.*}}, float %{{.*}})
extern struct homogeneous_struct struct_callee(struct homogeneous_struct);
struct homogeneous_struct test_struct(struct homogeneous_struct arg) {
  return struct_callee(arg);
}

struct nested_array {
  double d[4];
};
// CHECK: define arm_aapcs_vfpcc void @test_array(double %{{.*}}, double %{{.*}}, double %{{.*}}, double %{{.*}})
extern void array_callee(struct nested_array);
void test_array(struct nested_array arg) {
  array_callee(arg);
}

extern void complex_callee(__complex__ double);
// CHECK: define arm_aapcs_vfpcc void @test_complex(double %{{.*}}, double %{{.*}})
void test_complex(__complex__ double cd) {
  complex_callee(cd);
}

// Long double is the same as double on AAPCS, it should be homogeneous.
extern void complex_long_callee(__complex__ long double);
// CHECK: define arm_aapcs_vfpcc void @test_complex_long(double %{{.*}}, double %{{.*}})
void test_complex_long(__complex__ long double cd) {
  complex_callee(cd);
}

// Structs with more than 4 elements of the base type are not treated
// as homogeneous aggregates.  Test that.

struct big_struct {
  float f1;
  float f[2];
  float f3;
  float f4;
};
// CHECK: define arm_aapcs_vfpcc void @test_big([5 x i32] %{{.*}})
extern void big_callee(struct big_struct);
void test_big(struct big_struct arg) {
  big_callee(arg);
}

// Make sure that aggregates with multiple base types are not treated as
// homogeneous aggregates.

struct heterogeneous_struct {
  float f1;
  int i2;
};
// CHECK: define arm_aapcs_vfpcc void @test_hetero([2 x i32] %{{.*}})
extern void hetero_callee(struct heterogeneous_struct);
void test_hetero(struct heterogeneous_struct arg) {
  hetero_callee(arg);
}

// Neon multi-vector types are homogeneous aggregates.
// CHECK: define arm_aapcs_vfpcc <16 x i8> @f0(<16 x i8> %{{.*}}, <16 x i8> %{{.*}}, <16 x i8> %{{.*}}, <16 x i8> %{{.*}})
int8x16_t f0(int8x16x4_t v4) {
  return vaddq_s8(v4.val[0], v4.val[3]);
}

// ...and it doesn't matter whether the vectors are exactly the same, as long
// as they have the same size.

struct neon_struct {
  int8x8x2_t v12;
  int32x2_t v3;
  int16x4_t v4;
};
// CHECK: define arm_aapcs_vfpcc void @test_neon(<8 x i8> %{{.*}}, <8 x i8> %{{.*}}, <2 x i32> %{{.*}}, <4 x i16> %{{.*}})
extern void neon_callee(struct neon_struct);
void test_neon(struct neon_struct arg) {
  neon_callee(arg);
}

// CHECK: define arm_aapcs_vfpcc void @f33(%struct.s33* byval %s)
struct s33 { char buf[32*32]; };
void f33(struct s33 s) { }
