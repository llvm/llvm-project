// RUN: %clang_cc1 -triple arm64-apple-ios7 -target-abi darwinpcs -target-feature +sve -emit-llvm -o - %s | FileCheck %s
// RUN: %clang_cc1 -triple aarch64-linux-android -target-feature +sve -emit-llvm -o - %s | FileCheck %s

#include <stdarg.h>

#define SCALABLE_SIZE(N) (N), 1

typedef __attribute__(( ext_vector_type(SCALABLE_SIZE(1)) ))  char __char1s;
typedef __attribute__(( ext_vector_type(SCALABLE_SIZE(2)) ))  char __char2s;
typedef __attribute__(( ext_vector_type(SCALABLE_SIZE(3)) ))  char __char3s;
typedef __attribute__(( ext_vector_type(SCALABLE_SIZE(4)) ))  char __char4s;
typedef __attribute__(( ext_vector_type(SCALABLE_SIZE(1)) ))  short __short1s;
typedef __attribute__(( ext_vector_type(SCALABLE_SIZE(2)) ))  short __short2s;
typedef __attribute__(( ext_vector_type(SCALABLE_SIZE(3)) ))  short __short3s;
typedef __attribute__(( ext_vector_type(SCALABLE_SIZE(1)) ))  int __int1s;
typedef __attribute__(( ext_vector_type(SCALABLE_SIZE(4)) ))  int __int4s;
typedef __attribute__(( ext_vector_type(SCALABLE_SIZE(1)) ))  double __double1s;
typedef __attribute__(( ext_vector_type(SCALABLE_SIZE(2)) ))  double __double2s;

double svfunc__char1s(__char1s arg);

double vec_s1c(int fixed, __char1s c1s) {
// CHECK-LABEL: @vec_s1c
// CHECK: [[PTR:%.*]] = alloca <vscale x 16 x i8>, align 16
// CHECK: store <vscale x 16 x i8> %c1s, ptr [[PTR]], align 16
// CHECK: [[CALL:%.*]] = call double @svfunc__char1s(<vscale x 16 x i8> {{%.*}})
  double sum = fixed;

  return sum + svfunc__char1s(c1s);
}

double test_s1c(__char1s *in) {
// CHECK-LABEL: @test_s1c
// CHECK: call double @vec_s1c(i32 noundef 1, <vscale x 16 x i8> {{%.*}})
  return vec_s1c(1, *in);
}

double svfunc__char2s(__char2s arg);

double vec_s2c(int fixed, __char2s c2s) {
// CHECK-LABEL: @vec_s2c
// CHECK: [[PTR:%.*]] = alloca { <vscale x 16 x i8>, <vscale x 16 x i8> }, align 16
// CHECK: {{%.*}} = insertvalue {{.*}} poison, {{.*}}.coerce{{.*}}
// CHECK-COUNT-1: {{%.*}} = insertvalue {{.*}} {{%.*}}, {{.*}}.coerce{{.*}}
// CHECK: store { <vscale x 16 x i8>, <vscale x 16 x i8> } {{%.*}}, ptr [[PTR]], align 16
// CHECK: [[CALL:%.*]] = call double @svfunc__char2s(<vscale x 16 x i8> {{%.*}}.extract0, <vscale x 16 x i8> {{%.*}}.extract1)
  double sum = fixed;

  return sum + svfunc__char2s(c2s);
}

double test_s2c(__char2s *in) {
// CHECK-LABEL: @test_s2c
// CHECK: call double @vec_s2c(i32 noundef 1, <vscale x 16 x i8> {{%.*}}.extract0, <vscale x 16 x i8> {{%.*}}.extract1)
  return vec_s2c(1, *in);
}

double svfunc__char3s(__char3s arg);

double vec_s3c(int fixed, __char3s c3s) {
// CHECK-LABEL: @vec_s3c
// CHECK: [[PTR:%.*]] = alloca { <vscale x 16 x i8>, <vscale x 16 x i8>, <vscale x 16 x i8> }, align 16
// CHECK: {{%.*}} = insertvalue {{.*}} poison, {{.*}}.coerce{{.*}}
// CHECK-COUNT-2: {{%.*}} = insertvalue {{.*}} {{%.*}}, {{.*}}.coerce{{.*}}
// CHECK: store { <vscale x 16 x i8>, <vscale x 16 x i8>, <vscale x 16 x i8> } {{%.*}}, ptr [[PTR]], align 16
// CHECK: [[CALL:%.*]] = call double @svfunc__char3s(<vscale x 16 x i8> {{%.*}}.extract0, <vscale x 16 x i8> {{%.*}}.extract1, <vscale x 16 x i8> {{%.*}}.extract2)
  double sum = fixed;

  return sum + svfunc__char3s(c3s);
}

double test_s3c(__char3s *in) {
// CHECK-LABEL: @test_s3c
// CHECK: call double @vec_s3c(i32 noundef 1, <vscale x 16 x i8> {{%.*}}.extract0, <vscale x 16 x i8> {{%.*}}.extract1, <vscale x 16 x i8> {{%.*}}.extract2)
  return vec_s3c(1, *in);
}

double svfunc__char4s(__char4s arg);

double vec_s4c(int fixed, __char4s c4s) {
// CHECK-LABEL: @vec_s4c
// CHECK: [[PTR:%.*]] = alloca { <vscale x 16 x i8>, <vscale x 16 x i8>, <vscale x 16 x i8>, <vscale x 16 x i8> }, align 16
// CHECK: {{%.*}} = insertvalue {{.*}} poison, {{.*}}.coerce{{.*}}
// CHECK-COUNT-3: {{%.*}} = insertvalue {{.*}} {{%.*}}, {{.*}}.coerce{{.*}}
// CHECK: store { <vscale x 16 x i8>, <vscale x 16 x i8>, <vscale x 16 x i8>, <vscale x 16 x i8> } {{%.*}}, ptr [[PTR]], align 16
// CHECK: [[CALL:%.*]] = call double @svfunc__char4s(<vscale x 16 x i8> {{%.*}}.extract0, <vscale x 16 x i8> {{%.*}}.extract1, <vscale x 16 x i8> {{%.*}}.extract2, <vscale x 16 x i8> {{%.*}}.extract3)
  double sum = fixed;

  return sum + svfunc__char4s(c4s);
}

double test_s4c(__char4s *in) {
// CHECK-LABEL: @test_s4c
// CHECK: call double @vec_s4c(i32 noundef 1, <vscale x 16 x i8> {{%.*}}.extract0, <vscale x 16 x i8> {{%.*}}.extract1, <vscale x 16 x i8> {{%.*}}.extract2, <vscale x 16 x i8> {{%.*}}.extract3)
  return vec_s4c(1, *in);
}

double svfunc__short1s(__short1s arg);

double vec_s1s(int fixed, __short1s s1s) {
// CHECK-LABEL: @vec_s1s
// CHECK: [[PTR:%.*]] = alloca <vscale x 8 x i16>, align 16
// CHECK: store <vscale x 8 x i16> %s1s, ptr [[PTR]], align 16
// CHECK: [[CALL:%.*]] = call double @svfunc__short1s(<vscale x 8 x i16> {{%.*}})
  double sum = fixed;

  return sum + svfunc__short1s(s1s);
}

double test_s1s(__short1s *in) {
// CHECK-LABEL: @test_s1s
// CHECK: call double @vec_s1s(i32 noundef 1, <vscale x 8 x i16> {{%.*}})
  return vec_s1s(1, *in);
}

double svfunc__short2s(__short2s arg);

double vec_s2s(int fixed, __short2s s2s) {
// CHECK-LABEL: @vec_s2s
// CHECK: [[PTR:%.*]] = alloca { <vscale x 8 x i16>, <vscale x 8 x i16> }, align 16
// CHECK: {{%.*}} = insertvalue {{.*}} poison, {{.*}}.coerce{{.*}}
// CHECK-COUNT-1: {{%.*}} = insertvalue {{.*}} {{%.*}}, {{.*}}.coerce{{.*}}
// CHECK: store { <vscale x 8 x i16>, <vscale x 8 x i16> } {{%.*}}, ptr [[PTR]], align 16
// CHECK: [[CALL:%.*]] = call double @svfunc__short2s(<vscale x 8 x i16> {{%.*}}.extract0, <vscale x 8 x i16> {{%.*}}.extract1)
  double sum = fixed;

  return sum + svfunc__short2s(s2s);
}

double test_s2s(__short2s *in) {
// CHECK-LABEL: @test_s2s
// CHECK: call double @vec_s2s(i32 noundef 1, <vscale x 8 x i16> {{%.*}}.extract0, <vscale x 8 x i16> {{%.*}}.extract1)
  return vec_s2s(1, *in);
}

double svfunc__short3s(__short3s arg);

double vec_s3s(int fixed, __short3s s3s) {
// CHECK-LABEL: @vec_s3s
// CHECK: [[PTR:%.*]] = alloca { <vscale x 8 x i16>, <vscale x 8 x i16>, <vscale x 8 x i16> }, align 16
// CHECK: {{%.*}} = insertvalue {{.*}} poison, {{.*}}.coerce{{.*}}
// CHECK-COUNT-2: {{%.*}} = insertvalue {{.*}} {{%.*}}, {{.*}}.coerce{{.*}}
// CHECK: store { <vscale x 8 x i16>, <vscale x 8 x i16>, <vscale x 8 x i16> } {{%.*}}, ptr [[PTR]], align 16
// CHECK: [[CALL:%.*]] = call double @svfunc__short3s(<vscale x 8 x i16> {{%.*}}.extract0, <vscale x 8 x i16> {{%.*}}.extract1, <vscale x 8 x i16> {{%.*}}.extract2)
  double sum = fixed;

  return sum + svfunc__short3s(s3s);
}

double test_s3s(__short3s *in) {
// CHECK-LABEL: @test_s3s
// CHECK: call double @vec_s3s(i32 noundef 1, <vscale x 8 x i16> {{%.*}}.extract0, <vscale x 8 x i16> {{%.*}}.extract1, <vscale x 8 x i16> {{%.*}}.extract2)
  return vec_s3s(1, *in);
}

double svfunc__int1s(__int1s arg);

double vec_s1i(int fixed, __int1s i1s) {
// CHECK-LABEL: @vec_s1i
// CHECK: [[PTR:%.*]] = alloca <vscale x 4 x i32>, align 16
// CHECK: store <vscale x 4 x i32> %i1s, ptr [[PTR]], align 16
// CHECK: [[CALL:%.*]] = call double @svfunc__int1s(<vscale x 4 x i32> {{%.*}})
  double sum = fixed;

  return sum + svfunc__int1s(i1s);
}

double test_s1i(__int1s *in) {
// CHECK-LABEL: @test_s1i
// CHECK: call double @vec_s1i(i32 noundef 1, <vscale x 4 x i32> {{%.*}})
  return vec_s1i(1, *in);
}

double svfunc__int4s(__int4s arg);

double vec_s4i(int fixed, __int4s i4s) {
// CHECK-LABEL: @vec_s4i
// CHECK: [[PTR:%.*]] = alloca { <vscale x 4 x i32>, <vscale x 4 x i32>, <vscale x 4 x i32>, <vscale x 4 x i32> }, align 16
// CHECK: {{%.*}} = insertvalue {{.*}} poison, {{.*}}.coerce{{.*}}
// CHECK-COUNT-3: {{%.*}} = insertvalue {{.*}} {{%.*}}, {{.*}}.coerce{{.*}}
// CHECK: store { <vscale x 4 x i32>, <vscale x 4 x i32>, <vscale x 4 x i32>, <vscale x 4 x i32> } {{%.*}}, ptr [[PTR]], align 16
// CHECK: [[CALL:%.*]] = call double @svfunc__int4s(<vscale x 4 x i32> {{%.*}}.extract0, <vscale x 4 x i32> {{%.*}}.extract1, <vscale x 4 x i32> {{%.*}}.extract2, <vscale x 4 x i32> {{%.*}}.extract3)
  double sum = fixed;

  return sum + svfunc__int4s(i4s);
}

double test_s4i(__int4s *in) {
// CHECK-LABEL: @test_s4i
// CHECK: call double @vec_s4i(i32 noundef 1, <vscale x 4 x i32> {{%.*}}.extract0, <vscale x 4 x i32> {{%.*}}.extract1, <vscale x 4 x i32> {{%.*}}.extract2, <vscale x 4 x i32> {{%.*}}.extract3)
  return vec_s4i(1, *in);
}

double svfunc__double1s(__double1s arg);

double vec_s1d(int fixed, __double1s d1s) {
// CHECK-LABEL: @vec_s1d
// CHECK: [[PTR:%.*]] = alloca <vscale x 2 x double>, align 16
// CHECK: store <vscale x 2 x double> %d1s, ptr [[PTR]], align 16
// CHECK: [[CALL:%.*]] = call double @svfunc__double1s(<vscale x 2 x double> {{%.*}})
  double sum = fixed;

  return sum + svfunc__double1s(d1s);
}

double test_s1d(__double1s *in) {
// CHECK-LABEL: @test_s1d
// CHECK: call double @vec_s1d(i32 noundef 1, <vscale x 2 x double> {{%.*}})
  return vec_s1d(1, *in);
}

double svfunc__double2s(__double2s arg);

double vec_s2d(int fixed, __double2s d2s) {
// CHECK-LABEL: @vec_s2d
// CHECK: [[PTR:%.*]] = alloca { <vscale x 2 x double>, <vscale x 2 x double> }, align 16
// CHECK: {{%.*}} = insertvalue {{.*}} poison, {{.*}}.coerce{{.*}}
// CHECK-COUNT-1: {{%.*}} = insertvalue {{.*}} {{%.*}}, {{.*}}.coerce{{.*}}
// CHECK: store { <vscale x 2 x double>, <vscale x 2 x double> } {{%.*}}, ptr [[PTR]], align 16
// CHECK: [[CALL:%.*]] = call double @svfunc__double2s(<vscale x 2 x double> {{%.*}}.extract0, <vscale x 2 x double> {{%.*}}.extract1)
  double sum = fixed;

  return sum + svfunc__double2s(d2s);
}

double test_s2d(__double2s *in) {
// CHECK-LABEL: @test_s2d
// CHECK: call double @vec_s2d(i32 noundef 1, <vscale x 2 x double> {{%.*}}.extract0, <vscale x 2 x double> {{%.*}}.extract1)
  return vec_s2d(1, *in);
}
