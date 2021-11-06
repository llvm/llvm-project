// RUN: %clang_cc1 -triple arm64-apple-ios7 -target-abi darwinpcs -emit-llvm -o - %s | FileCheck %s
// RUN: %clang_cc1 -triple aarch64-linux-android -emit-llvm -o - %s | FileCheck -check-prefix=ANDROID %s

#include <stdarg.h>

typedef __attribute__(( ext_vector_type(2) ))  char __char2;
typedef __attribute__(( ext_vector_type(3) ))  char __char3;
typedef __attribute__(( ext_vector_type(4) ))  char __char4;
typedef __attribute__(( ext_vector_type(5) ))  char __char5;
typedef __attribute__(( ext_vector_type(9) ))  char __char9;
typedef __attribute__(( ext_vector_type(19) )) char __char19;
typedef __attribute__(( ext_vector_type(3) ))  short __short3;
typedef __attribute__(( ext_vector_type(5) ))  short __short5;
typedef __attribute__(( ext_vector_type(3) ))  int __int3;
typedef __attribute__(( ext_vector_type(5) ))  int __int5;
typedef __attribute__(( ext_vector_type(3) ))  double __double3;

// Passing legal vector types as varargs. Check that we've allocated the appropriate size
double varargs_vec_2c(int fixed, ...) {
// ANDROID: varargs_vec_2c
// ANDROID: [[VAR:%.*]] = alloca <2 x i8>, align 2
// ANDROID: [[AP_NEXT:%.*]] = getelementptr inbounds i8, i8* [[AP_CUR:%.*]], i64 8
// ANDROID: bitcast i8* [[AP_CUR]] to <2 x i8>*
  va_list ap;
  double sum = fixed;
  va_start(ap, fixed);
  __char2 c3 = va_arg(ap, __char2);
  sum = sum + c3.x + c3.y;
  va_end(ap);
  return sum;
}

double test_2c(__char2 *in) {
// ANDROID: call double (i32, ...) @varargs_vec_2c(i32 3, i16 {{%.*}})
  return varargs_vec_2c(3, *in);
}

double varargs_vec_3c(int fixed, ...) {
// CHECK: varargs_vec_3c
// CHECK: alloca <3 x i8>, align 4
// CHECK: [[AP_NEXT:%.*]] = getelementptr inbounds i8, i8* [[AP_CUR:%.*]], i64 8
// CHECK: bitcast i8* [[AP_CUR]] to <3 x i8>*
  va_list ap;
  double sum = fixed;
  va_start(ap, fixed);
  __char3 c3 = va_arg(ap, __char3);
  sum = sum + c3.x + c3.y;
  va_end(ap);
  return sum;
}

double test_3c(__char3 *in) {
// CHECK: test_3c
// CHECK: call double (i32, ...) @varargs_vec_3c(i32 3, i32 {{%.*}})
  return varargs_vec_3c(3, *in);
}

double varargs_vec_4c(int fixed, ...) {
// CHECK: varargs_vec_4c
// CHECK: alloca <4 x i8>, align 4
// CHECK: [[AP_NEXT:%.*]] = getelementptr inbounds i8, i8* [[AP_CUR:%.*]], i64 8
// CHECK: bitcast i8* [[AP_CUR]] to <4 x i8>*
  va_list ap;
  double sum = fixed;
  va_start(ap, fixed);
  __char4 c4 = va_arg(ap, __char4);
  sum = sum + c4.x + c4.y;
  va_end(ap);
  return sum;
}

double test_4c(__char4 *in) {
// CHECK: test_4c
// CHECK: call double (i32, ...) @varargs_vec_4c(i32 4, i32 {{%.*}})
  return varargs_vec_4c(4, *in);
}

double varargs_vec_5c(int fixed, ...) {
// CHECK: varargs_vec_5c
// CHECK: alloca <5 x i8>, align 8
// CHECK: [[AP_NEXT:%.*]] = getelementptr inbounds i8, i8* [[AP_CUR:%.*]], i64 8
// CHECK: bitcast i8* [[AP_CUR]] to <5 x i8>*
  va_list ap;
  double sum = fixed;
  va_start(ap, fixed);
  __char5 c5 = va_arg(ap, __char5);
  sum = sum + c5.x + c5.y;
  va_end(ap);
  return sum;
}

double test_5c(__char5 *in) {
// CHECK: test_5c
// CHECK: call double (i32, ...) @varargs_vec_5c(i32 5, <2 x i32> {{%.*}})
  return varargs_vec_5c(5, *in);
}

double varargs_vec_9c(int fixed, ...) {
// CHECK: varargs_vec_9c
// CHECK: alloca <9 x i8>, align 16
// CHECK: [[ALIGN:%.*]] = and i64 {{%.*}}, -16
// CHECK: [[AP_ALIGN:%.*]] = inttoptr i64 [[ALIGN]] to i8*
// CHECK: [[AP_NEXT:%.*]] = getelementptr inbounds i8, i8* [[AP_ALIGN]], i64 16
// CHECK: bitcast i8* [[AP_ALIGN]] to <9 x i8>*
  va_list ap;
  double sum = fixed;
  va_start(ap, fixed);
  __char9 c9 = va_arg(ap, __char9);
  sum = sum + c9.x + c9.y;
  va_end(ap);
  return sum;
}

double test_9c(__char9 *in) {
// CHECK: test_9c
// CHECK: call double (i32, ...) @varargs_vec_9c(i32 9, <4 x i32> {{%.*}})
  return varargs_vec_9c(9, *in);
}

double varargs_vec_19c(int fixed, ...) {
// CHECK: varargs_vec_19c
// CHECK: [[AP_NEXT:%.*]] = getelementptr inbounds i8, i8* [[AP_CUR:%.*]], i64 8
// CHECK: [[VAR:%.*]] = bitcast i8* [[AP_CUR]] to <19 x i8>**
// CHECK: [[VAR2:%.*]] = load <19 x i8>*, <19 x i8>** [[VAR]]
  va_list ap;
  double sum = fixed;
  va_start(ap, fixed);
  __char19 c19 = va_arg(ap, __char19);
  sum = sum + c19.x + c19.y;
  va_end(ap);
  return sum;
}

double test_19c(__char19 *in) {
// CHECK: test_19c
// CHECK: call double (i32, ...) @varargs_vec_19c(i32 19, <19 x i8>* {{%.*}})
  return varargs_vec_19c(19, *in);
}

double varargs_vec_3s(int fixed, ...) {
// CHECK: varargs_vec_3s
// CHECK: alloca <3 x i16>, align 8
// CHECK: [[AP_NEXT:%.*]] = getelementptr inbounds i8, i8* [[AP_CUR:%.*]], i64 8
// CHECK: bitcast i8* [[AP_CUR]] to <3 x i16>*
  va_list ap;
  double sum = fixed;
  va_start(ap, fixed);
  __short3 c3 = va_arg(ap, __short3);
  sum = sum + c3.x + c3.y;
  va_end(ap);
  return sum;
}

double test_3s(__short3 *in) {
// CHECK: test_3s
// CHECK: call double (i32, ...) @varargs_vec_3s(i32 3, <2 x i32> {{%.*}})
  return varargs_vec_3s(3, *in);
}

double varargs_vec_5s(int fixed, ...) {
// CHECK: varargs_vec_5s
// CHECK: alloca <5 x i16>, align 16
// CHECK: [[ALIGN:%.*]] = and i64 {{%.*}}, -16
// CHECK: [[AP_ALIGN:%.*]] = inttoptr i64 [[ALIGN]] to i8*
// CHECK: [[AP_NEXT:%.*]] = getelementptr inbounds i8, i8* [[AP_ALIGN]], i64 16
// CHECK: bitcast i8* [[AP_ALIGN]] to <5 x i16>*
  va_list ap;
  double sum = fixed;
  va_start(ap, fixed);
  __short5 c5 = va_arg(ap, __short5);
  sum = sum + c5.x + c5.y;
  va_end(ap);
  return sum;
}

double test_5s(__short5 *in) {
// CHECK: test_5s
// CHECK: call double (i32, ...) @varargs_vec_5s(i32 5, <4 x i32> {{%.*}})
  return varargs_vec_5s(5, *in);
}

double varargs_vec_3i(int fixed, ...) {
// CHECK: varargs_vec_3i
// CHECK: alloca <3 x i32>, align 16
// CHECK: [[ALIGN:%.*]] = and i64 {{%.*}}, -16
// CHECK: [[AP_ALIGN:%.*]] = inttoptr i64 [[ALIGN]] to i8*
// CHECK: [[AP_NEXT:%.*]] = getelementptr inbounds i8, i8* [[AP_ALIGN]], i64 16
// CHECK: bitcast i8* [[AP_ALIGN]] to <3 x i32>*
  va_list ap;
  double sum = fixed;
  va_start(ap, fixed);
  __int3 c3 = va_arg(ap, __int3);
  sum = sum + c3.x + c3.y;
  va_end(ap);
  return sum;
}

double test_3i(__int3 *in) {
// CHECK: test_3i
// CHECK: call double (i32, ...) @varargs_vec_3i(i32 3, <4 x i32> {{%.*}})
  return varargs_vec_3i(3, *in);
}

double varargs_vec_5i(int fixed, ...) {
// CHECK: varargs_vec_5i
// CHECK: alloca <5 x i32>, align 16
// CHECK: [[AP_NEXT:%.*]] = getelementptr inbounds i8, i8* [[AP_CUR:%.*]], i64 8
// CHECK: [[VAR:%.*]] = bitcast i8* [[AP_CUR]] to <5 x i32>**
// CHECK: [[VAR2:%.*]] = load <5 x i32>*, <5 x i32>** [[VAR]]
  va_list ap;
  double sum = fixed;
  va_start(ap, fixed);
  __int5 c5 = va_arg(ap, __int5);
  sum = sum + c5.x + c5.y;
  va_end(ap);
  return sum;
}

double test_5i(__int5 *in) {
// CHECK: test_5i
// CHECK: call double (i32, ...) @varargs_vec_5i(i32 5, <5 x i32>* {{%.*}})
  return varargs_vec_5i(5, *in);
}

double varargs_vec_3d(int fixed, ...) {
// CHECK: varargs_vec_3d
// CHECK: alloca <3 x double>, align 16
// CHECK: [[AP_NEXT:%.*]] = getelementptr inbounds i8, i8* [[AP_CUR:%.*]], i64 8
// CHECK: [[VAR:%.*]] = bitcast i8* [[AP_CUR]] to <3 x double>**
// CHECK: [[VAR2:%.*]] = load <3 x double>*, <3 x double>** [[VAR]]
  va_list ap;
  double sum = fixed;
  va_start(ap, fixed);
  __double3 c3 = va_arg(ap, __double3);
  sum = sum + c3.x + c3.y;
  va_end(ap);
  return sum;
}

double test_3d(__double3 *in) {
// CHECK: test_3d
// CHECK: call double (i32, ...) @varargs_vec_3d(i32 3, <3 x double>* {{%.*}})
  return varargs_vec_3d(3, *in);
}

double varargs_vec(int fixed, ...) {
// CHECK: varargs_vec
  va_list ap;
  double sum = fixed;
  va_start(ap, fixed);
  __char3 c3 = va_arg(ap, __char3);
// CHECK: [[AP_NEXT:%.*]] = getelementptr inbounds i8, i8* [[AP_CUR:%.*]], i64 8
// CHECK: bitcast i8* [[AP_CUR]] to <3 x i8>*
  sum = sum + c3.x + c3.y;
  __char5 c5 = va_arg(ap, __char5);
// CHECK: [[AP_NEXT:%.*]] = getelementptr inbounds i8, i8* [[AP_CUR:%.*]], i64 8
// CHECK: bitcast i8* [[AP_CUR]] to <5 x i8>*
  sum = sum + c5.x + c5.y;
  __char9 c9 = va_arg(ap, __char9);
// CHECK: [[ALIGN:%.*]] = and i64 {{%.*}}, -16
// CHECK: [[AP_ALIGN:%.*]] = inttoptr i64 [[ALIGN]] to i8*
// CHECK: [[AP_NEXT:%.*]] = getelementptr inbounds i8, i8* [[AP_ALIGN]], i64 16
// CHECK: bitcast i8* [[AP_ALIGN]] to <9 x i8>*
  sum = sum + c9.x + c9.y;
  __char19 c19 = va_arg(ap, __char19);
// CHECK: [[AP_NEXT:%.*]] = getelementptr inbounds i8, i8* [[AP_CUR:%.*]], i64 8
// CHECK: [[VAR:%.*]] = bitcast i8* [[AP_CUR]] to <19 x i8>**
// CHECK: [[VAR2:%.*]] = load <19 x i8>*, <19 x i8>** [[VAR]]
  sum = sum + c19.x + c19.y;
  __short3 s3 = va_arg(ap, __short3);
// CHECK: [[AP_NEXT:%.*]] = getelementptr inbounds i8, i8* [[AP_CUR:%.*]], i64 8
// CHECK: bitcast i8* [[AP_CUR]] to <3 x i16>*
  sum = sum + s3.x + s3.y;
  __short5 s5 = va_arg(ap, __short5);
// CHECK: [[ALIGN:%.*]] = and i64 {{%.*}}, -16
// CHECK: [[AP_ALIGN:%.*]] = inttoptr i64 [[ALIGN]] to i8*
// CHECK: [[AP_NEXT:%.*]] = getelementptr inbounds i8, i8* [[AP_ALIGN]], i64 16
// CHECK: bitcast i8* [[AP_ALIGN]] to <5 x i16>*
  sum = sum + s5.x + s5.y;
  __int3 i3 = va_arg(ap, __int3);
// CHECK: [[ALIGN:%.*]] = and i64 {{%.*}}, -16
// CHECK: [[AP_ALIGN:%.*]] = inttoptr i64 [[ALIGN]] to i8*
// CHECK: [[AP_NEXT:%.*]] = getelementptr inbounds i8, i8* [[AP_ALIGN]], i64 16
// CHECK: bitcast i8* [[AP_ALIGN]] to <3 x i32>*
  sum = sum + i3.x + i3.y;
  __int5 i5 = va_arg(ap, __int5);
// CHECK: [[AP_NEXT:%.*]] = getelementptr inbounds i8, i8* [[AP_CUR:%.*]], i64 8
// CHECK: [[VAR:%.*]] = bitcast i8* [[AP_CUR]] to <5 x i32>**
// CHECK: [[VAR2:%.*]] = load <5 x i32>*, <5 x i32>** [[VAR]]
  sum = sum + i5.x + i5.y;
  __double3 d3 = va_arg(ap, __double3);
// CHECK: [[AP_NEXT:%.*]] = getelementptr inbounds i8, i8* [[AP_CUR:%.*]], i64 8
// CHECK: [[VAR:%.*]] = bitcast i8* [[AP_CUR]] to <3 x double>**
// CHECK: [[VAR2:%.*]] = load <3 x double>*, <3 x double>** [[VAR]]
  sum = sum + d3.x + d3.y;
  va_end(ap);
  return sum;
}

double test(__char3 *c3, __char5 *c5, __char9 *c9, __char19 *c19,
            __short3 *s3, __short5 *s5, __int3 *i3, __int5 *i5,
            __double3 *d3) {
  double ret = varargs_vec(3, *c3, *c5, *c9, *c19, *s3, *s5, *i3, *i5, *d3);
// CHECK: call double (i32, ...) @varargs_vec(i32 3, i32 {{%.*}}, <2 x i32> {{%.*}}, <4 x i32> {{%.*}}, <19 x i8>* {{%.*}}, <2 x i32> {{%.*}}, <4 x i32> {{%.*}}, <4 x i32> {{%.*}}, <5 x i32>* {{%.*}}, <3 x double>* {{%.*}})
  return ret;
}

__attribute__((noinline)) double args_vec_3c(int fixed, __char3 c3) {
// CHECK: args_vec_3c
// CHECK: [[C3:%.*]] = alloca <3 x i8>, align 4
// CHECK: [[TMP:%.*]] = bitcast <3 x i8>* [[C3]] to i32*
// CHECK: store i32 {{%.*}}, i32* [[TMP]]
  double sum = fixed;
  sum = sum + c3.x + c3.y;
  return sum;
}

double fixed_3c(__char3 *in) {
// CHECK: fixed_3c
// CHECK: call double @args_vec_3c(i32 3, i32 {{%.*}})
  return args_vec_3c(3, *in);
}

__attribute__((noinline)) double args_vec_5c(int fixed, __char5 c5) {
// CHECK: args_vec_5c
// CHECK: [[C5:%.*]] = alloca <5 x i8>, align 8
// CHECK: [[TMP:%.*]] = bitcast <5 x i8>* [[C5]] to <2 x i32>*
// CHECK: store <2 x i32> {{%.*}}, <2 x i32>* [[TMP]], align 8
  double sum = fixed;
  sum = sum + c5.x + c5.y;
  return sum;
}

double fixed_5c(__char5 *in) {
// CHECK: fixed_5c
// CHECK: call double @args_vec_5c(i32 5, <2 x i32> {{%.*}})
  return args_vec_5c(5, *in);
}

__attribute__((noinline)) double args_vec_9c(int fixed, __char9 c9) {
// CHECK: args_vec_9c
// CHECK: [[C9:%.*]] = alloca <9 x i8>, align 16
// CHECK: [[TMP:%.*]] = bitcast <9 x i8>* [[C9]] to <4 x i32>*
// CHECK: store <4 x i32> {{%.*}}, <4 x i32>* [[TMP]], align 16
  double sum = fixed;
  sum = sum + c9.x + c9.y;
  return sum;
}

double fixed_9c(__char9 *in) {
// CHECK: fixed_9c
// CHECK: call double @args_vec_9c(i32 9, <4 x i32> {{%.*}})
  return args_vec_9c(9, *in);
}

__attribute__((noinline)) double args_vec_19c(int fixed, __char19 c19) {
// CHECK: args_vec_19c
// CHECK: [[C19:%.*]] = load <19 x i8>, <19 x i8>* {{.*}}, align 16
  double sum = fixed;
  sum = sum + c19.x + c19.y;
  return sum;
}

double fixed_19c(__char19 *in) {
// CHECK: fixed_19c
// CHECK: call double @args_vec_19c(i32 19, <19 x i8>* {{%.*}})
  return args_vec_19c(19, *in);
}

__attribute__((noinline)) double args_vec_3s(int fixed, __short3 c3) {
// CHECK: args_vec_3s
// CHECK: [[C3:%.*]] = alloca <3 x i16>, align 8
// CHECK: [[TMP:%.*]] = bitcast <3 x i16>* [[C3]] to <2 x i32>*
// CHECK: store <2 x i32> {{%.*}}, <2 x i32>* [[TMP]], align 8
  double sum = fixed;
  sum = sum + c3.x + c3.y;
  return sum;
}

double fixed_3s(__short3 *in) {
// CHECK: fixed_3s
// CHECK: call double @args_vec_3s(i32 3, <2 x i32> {{%.*}})
  return args_vec_3s(3, *in);
}

__attribute__((noinline)) double args_vec_5s(int fixed, __short5 c5) {
// CHECK: args_vec_5s
// CHECK: [[C5:%.*]] = alloca <5 x i16>, align 16
// CHECK: [[TMP:%.*]] = bitcast <5 x i16>* [[C5]] to <4 x i32>*
// CHECK: store <4 x i32> {{%.*}}, <4 x i32>* [[TMP]], align 16
  double sum = fixed;
  sum = sum + c5.x + c5.y;
  return sum;
}

double fixed_5s(__short5 *in) {
// CHECK: fixed_5s
// CHECK: call double @args_vec_5s(i32 5, <4 x i32> {{%.*}})
  return args_vec_5s(5, *in);
}

__attribute__((noinline)) double args_vec_3i(int fixed, __int3 c3) {
// CHECK: args_vec_3i
// CHECK: [[C3:%.*]] = alloca <3 x i32>, align 16
// CHECK: [[TMP:%.*]] = bitcast <3 x i32>* [[C3]] to <4 x i32>*
// CHECK: store <4 x i32> {{%.*}}, <4 x i32>* [[TMP]], align 16
  double sum = fixed;
  sum = sum + c3.x + c3.y;
  return sum;
}

double fixed_3i(__int3 *in) {
// CHECK: fixed_3i
// CHECK: call double @args_vec_3i(i32 3, <4 x i32> {{%.*}})
  return args_vec_3i(3, *in);
}

__attribute__((noinline)) double args_vec_5i(int fixed, __int5 c5) {
// CHECK: args_vec_5i
// CHECK: [[C5:%.*]] = load <5 x i32>, <5 x i32>* {{%.*}}, align 16
  double sum = fixed;
  sum = sum + c5.x + c5.y;
  return sum;
}

double fixed_5i(__int5 *in) {
// CHECK: fixed_5i
// CHECK: call double @args_vec_5i(i32 5, <5 x i32>* {{%.*}})
  return args_vec_5i(5, *in);
}

__attribute__((noinline)) double args_vec_3d(int fixed, __double3 c3) {
  // CHECK: args_vec_3d
  // CHECK: [[CAST:%.*]] = bitcast <3 x double>* {{%.*}} to <4 x double>*
  // CHECK: [[LOAD:%.*]] = load <4 x double>, <4 x double>* [[CAST]]
  // CHECK: shufflevector <4 x double> [[LOAD]], <4 x double> poison, <3 x i32> <i32 0, i32 1, i32 2>
  double sum = fixed;
  sum = sum + c3.x + c3.y;
  return sum;
}

double fixed_3d(__double3 *in) {
// CHECK: fixed_3d
// CHECK: call double @args_vec_3d(i32 3, <3 x double>* {{%.*}})
  return args_vec_3d(3, *in);
}
