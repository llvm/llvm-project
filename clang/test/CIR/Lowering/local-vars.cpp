// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck -input-file=%t-cir.ll %s

void test() {
  int i = 1;
  long l = 2l;
  float f = 3.0f;
  double d = 4.0;
  bool b1 = true;
  bool b2 = false;
  const int ci = 1;
  const long cl = 2l;
  const float cf = 3.0f;
  const double cd = 4.0;
  const bool cb1 = true;
  const bool cb2 = false;
  int uii;
  long uil;
  float uif;
  double uid;
  bool uib;
}

// Note: The alignment of i64 stores below is wrong. That should be fixed
//       when we add alignment attributes to the load/store ops.

// CHECK: define{{.*}} void @_Z4testv()
// CHECK:    %[[I_PTR:.*]] = alloca i32, i64 1, align 4
// CHECK:    %[[L_PTR:.*]] = alloca i64, i64 1, align 8
// CHECK:    %[[F_PTR:.*]] = alloca float, i64 1, align 4
// CHECK:    %[[D_PTR:.*]] = alloca double, i64 1, align 8
// CHECK:    %[[B1_PTR:.*]] = alloca i8, i64 1, align 1
// CHECK:    %[[B2_PTR:.*]] = alloca i8, i64 1, align 1
// CHECK:    %[[CI_PTR:.*]] = alloca i32, i64 1, align 4
// CHECK:    %[[CL_PTR:.*]] = alloca i64, i64 1, align 8
// CHECK:    %[[CF_PTR:.*]] = alloca float, i64 1, align 4
// CHECK:    %[[CD_PTR:.*]] = alloca double, i64 1, align 8
// CHECK:    %[[CB1_PTR:.*]] = alloca i8, i64 1, align 1
// CHECK:    %[[CB2_PTR:.*]] = alloca i8, i64 1, align 1
// CHECK:    %[[UII_PTR:.*]] = alloca i32, i64 1, align 4
// CHECK:    %[[UIL_PTR:.*]] = alloca i64, i64 1, align 8
// CHECK:    %[[UIF_PTR:.*]] = alloca float, i64 1, align 4
// CHECK:    %[[UID_PTR:.*]] = alloca double, i64 1, align 8
// CHECK:    %[[UIB_PTR:.*]] = alloca i8, i64 1, align 1
// CHECK:    store i32 1, ptr %[[I_PTR]], align 4
// CHECK:    store i64 2, ptr %[[L_PTR]], align 8
// CHECK:    store float 3.000000e+00, ptr %[[F_PTR]], align 4
// CHECK:    store double 4.000000e+00, ptr %[[D_PTR]], align 8
// CHECK:    store i8 1, ptr %[[B1_PTR]], align 1
// CHECK:    store i8 0, ptr %[[B2_PTR]], align 1
// CHECK:    store i32 1, ptr %[[CI_PTR]], align 4
// CHECK:    store i64 2, ptr %[[CL_PTR]], align 8
// CHECK:    store float 3.000000e+00, ptr %[[CF_PTR]], align 4
// CHECK:    store double 4.000000e+00, ptr %[[CD_PTR]], align 8
// CHECK:    store i8 1, ptr %[[CB1_PTR]], align 1
// CHECK:    store i8 0, ptr %[[CB2_PTR]], align 1
