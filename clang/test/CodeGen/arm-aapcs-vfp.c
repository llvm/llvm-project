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

// RUN: %clang_cc1 -triple arm64-apple-darwin9 -target-feature +neon \
// RUN:   -ffreestanding \
// RUN:   -emit-llvm -w -o - %s | FileCheck -check-prefix=CHECK64 %s

// REQUIRES: arm-registered-target
// REQUIRES: aarch64-registered-target

#include <arm_neon.h>

struct homogeneous_struct {
  float f[2];
  float f3;
  float f4;
};
// CHECK: define{{.*}} arm_aapcs_vfpcc %struct.homogeneous_struct @test_struct(%struct.homogeneous_struct %{{.*}})
// CHECK64: define{{.*}} %struct.homogeneous_struct @test_struct([4 x float] %{{.*}})
extern struct homogeneous_struct struct_callee(struct homogeneous_struct);
struct homogeneous_struct test_struct(struct homogeneous_struct arg) {
  return struct_callee(arg);
}

// CHECK: define{{.*}} arm_aapcs_vfpcc void @test_struct_variadic(%struct.homogeneous_struct* {{.*}}, ...)
struct homogeneous_struct test_struct_variadic(struct homogeneous_struct arg, ...) {
  return struct_callee(arg);
}

struct nested_array {
  double d[4];
};
// CHECK: define{{.*}} arm_aapcs_vfpcc void @test_array(%struct.nested_array %{{.*}})
// CHECK64: define{{.*}} void @test_array([4 x double] %{{.*}})
extern void array_callee(struct nested_array);
void test_array(struct nested_array arg) {
  array_callee(arg);
}

extern void complex_callee(__complex__ double);
// CHECK: define{{.*}} arm_aapcs_vfpcc void @test_complex({ double, double } %{{.*}})
// CHECK64: define{{.*}} void @test_complex([2 x double] %cd.coerce)
void test_complex(__complex__ double cd) {
  complex_callee(cd);
}

// Long double is the same as double on AAPCS, it should be homogeneous.
extern void complex_long_callee(__complex__ long double);
// CHECK: define{{.*}} arm_aapcs_vfpcc void @test_complex_long({ double, double } %{{.*}})
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
// CHECK: define{{.*}} arm_aapcs_vfpcc void @test_big([5 x i32] %{{.*}})
// CHECK64: define{{.*}} void @test_big(%struct.big_struct* %{{.*}})
// CHECK64: call void @llvm.memcpy
// CHECK64: call void @big_callee(%struct.big_struct*
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
// CHECK: define{{.*}} arm_aapcs_vfpcc void @test_hetero([2 x i32] %{{.*}})
// CHECK64: define{{.*}} void @test_hetero(i64 %{{.*}})
extern void hetero_callee(struct heterogeneous_struct);
void test_hetero(struct heterogeneous_struct arg) {
  hetero_callee(arg);
}

// Neon multi-vector types are homogeneous aggregates.
// CHECK: define{{.*}} arm_aapcs_vfpcc <16 x i8> @f0(%struct.int8x16x4_t %{{.*}})
// CHECK64: define{{.*}} <16 x i8> @f0([4 x <16 x i8>] %{{.*}})
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
// CHECK: define{{.*}} arm_aapcs_vfpcc void @test_neon(%struct.neon_struct %{{.*}})
// CHECK64: define{{.*}} void @test_neon([4 x <8 x i8>] %{{.*}})
extern void neon_callee(struct neon_struct);
void test_neon(struct neon_struct arg) {
  neon_callee(arg);
}

// CHECK-LABEL: define{{.*}} arm_aapcs_vfpcc void @f33(%struct.s33* byval(%struct.s33) align 4 %s)
struct s33 { char buf[32*32]; };
void f33(struct s33 s) { }

typedef struct { long long x; int y; } struct_long_long_int;
// CHECK: define{{.*}} arm_aapcs_vfpcc void @test_vfp_stack_gpr_split_1(double %a, double %b, double %c, double %d, double %e, double %f, double %g, double %h, double %i, i32 %j, i64 %k, i32 %l)
void test_vfp_stack_gpr_split_1(double a, double b, double c, double d, double e, double f, double g, double h, double i, int j, long long k, int l) {}

// CHECK: define{{.*}} arm_aapcs_vfpcc void @test_vfp_stack_gpr_split_2(double %a, double %b, double %c, double %d, double %e, double %f, double %g, double %h, double %i, i32 %j, [2 x i64] %k.coerce)
void test_vfp_stack_gpr_split_2(double a, double b, double c, double d, double e, double f, double g, double h, double i, int j, struct_long_long_int k) {}

// CHECK: define{{.*}} arm_aapcs_vfpcc void @test_vfp_stack_gpr_split_3(%struct.struct_long_long_int* noalias sret(%struct.struct_long_long_int) align 8 %agg.result, double %a, double %b, double %c, double %d, double %e, double %f, double %g, double %h, double %i, [2 x i64] %k.coerce)
struct_long_long_int test_vfp_stack_gpr_split_3(double a, double b, double c, double d, double e, double f, double g, double h, double i, struct_long_long_int k) {}

typedef struct { int a; int b:4; int c; } struct_int_bitfield_int;
// CHECK: define{{.*}} arm_aapcs_vfpcc void @test_test_vfp_stack_gpr_split_bitfield(double %a, double %b, double %c, double %d, double %e, double %f, double %g, double %h, double %i, i32 %j, i32 %k, [3 x i32] %l.coerce)
void test_test_vfp_stack_gpr_split_bitfield(double a, double b, double c, double d, double e, double f, double g, double h, double i, int j, int k, struct_int_bitfield_int l) {}

// Note: this struct requires internal padding
typedef struct { int x; long long y; } struct_int_long_long;
// CHECK: define{{.*}} arm_aapcs_vfpcc void @test_vfp_stack_gpr_split_4(double %a, double %b, double %c, double %d, double %e, double %f, double %g, double %h, double %i, i32 %j, [2 x i64] %k.coerce)
void test_vfp_stack_gpr_split_4(double a, double b, double c, double d, double e, double f, double g, double h, double i, int j, struct_int_long_long k) {}

// This very large struct (passed byval) uses up the GPRs, so no padding is needed
typedef struct { int x[17]; } struct_seventeen_ints;
typedef struct { int x[4]; } struct_four_ints;
// CHECK: define{{.*}} arm_aapcs_vfpcc void @test_vfp_stack_gpr_split_5(%struct.struct_seventeen_ints* byval(%struct.struct_seventeen_ints) align 4 %a, double %b, double %c, double %d, double %e, double %f, double %g, double %h, double %i, double %j, [4 x i32] %k.coerce)
void test_vfp_stack_gpr_split_5(struct_seventeen_ints a, double b, double c, double d, double e, double f, double g, double h, double i, double j, struct_four_ints k) {}

// Here, parameter k would need padding to prevent it from being split, but it
// is passed ByVal (due to being > 64 bytes), so the backend handles this instead.
void test_vfp_stack_gpr_split_6(double a, double b, double c, double d, double e, double f, double g, double h, double i, int j, struct_seventeen_ints k) {}
// CHECK: define{{.*}} arm_aapcs_vfpcc void @test_vfp_stack_gpr_split_6(double %a, double %b, double %c, double %d, double %e, double %f, double %g, double %h, double %i, i32 %j, %struct.struct_seventeen_ints* byval(%struct.struct_seventeen_ints) align 4 %k)
