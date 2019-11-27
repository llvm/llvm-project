// RUN: %clang_cc1 -triple armv7a--none-eabi -target-abi aapcs \
// RUN:   -mfloat-abi soft -target-feature +neon -emit-llvm -o - -O2 %s \
// RUN:   | FileCheck %s --check-prefix=CHECK-SOFT
// RUN: %clang_cc1 -triple armv7a--none-eabi -target-abi aapcs \
// RUN:   -mfloat-abi hard -target-feature +neon -emit-llvm -o - -O2 %s \
// RUN:   | FileCheck %s --check-prefix=CHECK-HARD
// RUN: %clang_cc1 -triple armv7a--none-eabi -target-abi aapcs \
// RUN:   -mfloat-abi hard -target-feature +neon -target-feature +fullfp16 \
// RUN:   -emit-llvm -o - -O2 %s \
// RUN:   | FileCheck %s --check-prefix=CHECK-FULL

typedef float float32_t;
typedef __fp16 float16_t;
typedef __attribute__((neon_vector_type(2))) float32_t float32x2_t;
typedef __attribute__((neon_vector_type(4))) float16_t float16x4_t;

struct S1 {
  float32x2_t M1;
  float16x4_t M2;
};

struct B1 { float32x2_t M; };
struct B2 { float16x4_t M; };

struct S2 : B1, B2 {};

struct S3 : B1 {
  float16x4_t M;
};

struct S4 : B1 {
  B2 M[1];
};

// S5 does not contain any FP16 vectors
struct S5 : B1 {
  B1 M[1];
};

// CHECK-SOFT: define void @_Z2f12S1(%struct.S1* noalias nocapture sret %agg.result, [2 x i64] %s1.coerce)
// CHECK-HARD: define arm_aapcs_vfpcc [2 x <2 x i32>] @_Z2f12S1([2 x <2 x i32>] returned %s1.coerce)
// CHECK-FULL: define arm_aapcs_vfpcc %struct.S1 @_Z2f12S1(%struct.S1 returned %s1.coerce)
struct S1 f1(struct S1 s1) { return s1; }

// CHECK-SOFT: define void @_Z2f22S2(%struct.S2* noalias nocapture sret %agg.result, [4 x i32] %s2.coerce)
// CHECK-HARD: define arm_aapcs_vfpcc [2 x <2 x i32>] @_Z2f22S2([2 x <2 x i32>] returned %s2.coerce)
// CHECK-FULL: define arm_aapcs_vfpcc %struct.S2 @_Z2f22S2(%struct.S2 returned %s2.coerce)
struct S2 f2(struct S2 s2) { return s2; }

// CHECK-SOFT: define void @_Z2f32S3(%struct.S3* noalias nocapture sret %agg.result, [2 x i64] %s3.coerce)
// CHECK-HARD: define arm_aapcs_vfpcc [2 x <2 x i32>] @_Z2f32S3([2 x <2 x i32>] returned %s3.coerce)
// CHECK-FULL: define arm_aapcs_vfpcc %struct.S3 @_Z2f32S3(%struct.S3 returned %s3.coerce)
struct S3 f3(struct S3 s3) { return s3; }

// CHECK-SOFT: define void @_Z2f42S4(%struct.S4* noalias nocapture sret %agg.result, [2 x i64] %s4.coerce)
// CHECK-HARD: define arm_aapcs_vfpcc [2 x <2 x i32>] @_Z2f42S4([2 x <2 x i32>] returned %s4.coerce)
// CHECK-FULL: define arm_aapcs_vfpcc %struct.S4 @_Z2f42S4(%struct.S4 returned %s4.coerce)
struct S4 f4(struct S4 s4) { return s4; }

// CHECK-SOFT: define void @_Z2f52S5(%struct.S5* noalias nocapture sret %agg.result, [2 x i64] %s5.coerce)
// CHECK-HARD: define arm_aapcs_vfpcc %struct.S5 @_Z2f52S5(%struct.S5 returned %s5.coerce)
// CHECK-FULL: define arm_aapcs_vfpcc %struct.S5 @_Z2f52S5(%struct.S5 returned %s5.coerce)
struct S5 f5(struct S5 s5) { return s5; }
