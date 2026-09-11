// RUN: %clang_cc1 -triple x86_64-apple-darwin9 -target-cpu corei7-avx -emit-llvm %s -o - | FileCheck %s
// RUN: %clang_cc1 -triple x86_64-apple-darwin9 -target-cpu corei7-avx -emit-llvm -x c++ %s -o - | FileCheck %s

#include <stdbool.h>

typedef double vector8double __attribute__((__vector_size__(64)));
typedef float  vector8float  __attribute__((__vector_size__(32)));
typedef bool   vector8bool   __attribute__((__ext_vector_type__(8)));
typedef long   vector8long   __attribute__((__vector_size__(64)));
typedef unsigned long   vector8ulong   __attribute__((__vector_size__(64)));

#ifdef __cplusplus
extern "C" {
#endif

vector8bool splat_bool_imm_vector8bool(void) {
  return __builtin_splatvector(true, vector8bool);
  // CHECK-LABEL: @splat_bool_imm_vector8bool
  // CHECK: store <8 x i1> splat (i1 true), ptr %[[RETPTR:.*]]
}

vector8bool splat_int_imm_vector8bool(void) {
  return __builtin_splatvector(-1, vector8bool);
  // CHECK-LABEL: @splat_int_imm_vector8bool
  // CHECK: store <8 x i1> splat (i1 true), ptr %[[RETPTR:.*]]
}

vector8bool splat_uint_imm_vector8bool(void) {
  return __builtin_splatvector(1U, vector8bool);
  // CHECK-LABEL: @splat_uint_imm_vector8bool
  // CHECK: store <8 x i1> splat (i1 true), ptr %[[RETPTR:.*]]
}

vector8bool splat_float_imm_vector8bool(void) {
  return __builtin_splatvector(1.0f, vector8bool);
  // CHECK-LABEL: @splat_float_imm_vector8bool
  // CHECK: store <8 x i1> splat (i1 true), ptr %[[RETPTR:.*]]
}

vector8bool splat_double_imm_vector8bool(void) {
  return __builtin_splatvector(1.0, vector8bool);
  // CHECK-LABEL: @splat_double_imm_vector8bool
  // CHECK: store <8 x i1> splat (i1 true), ptr %[[RETPTR:.*]]
}

vector8bool splat_bool_var_vector8bool(bool x) {
  return __builtin_splatvector(x, vector8bool);
  // CHECK-LABEL: @splat_bool_var_vector8bool
  // CHECK: %[[STOREDV:.*]] = zext i1 %x to i8
  // CHECK: store i8 %[[STOREDV]], ptr %[[WHERE:.*]]
  // CHECK: %[[WHAT:.*]] = load i8, ptr %[[WHERE]]
  // CHECK: %[[WHAT_I1:.*]] = icmp ne i8 %[[WHAT]], 0
  // CHECK: %[[INSERT:.*]] = insertelement <8 x i1> poison, i1 %[[WHAT_I1]], i64 0
  // CHECK: %[[SPLAT:.*]] = shufflevector <8 x i1> %[[INSERT]], <8 x i1> poison, <8 x i32> zeroinitializer
  // CHECK: store <8 x i1> %[[SPLAT]], ptr %[[RETPTR:.*]]
}

vector8bool splat_int_var_vector8bool(int x) {
  return __builtin_splatvector(x, vector8bool);
  // CHECK-LABEL: @splat_int_var_vector8bool
  // CHECK: store i32 %x, ptr %[[WHERE:.*]]
  // CHECK: %[[WHAT:.*]] = load i32, ptr %[[WHERE]]
  // CHECK: %[[INSERT:.*]] = insertelement <8 x i32> poison, i32 %[[WHAT]], i64 0
  // CHECK: %[[SPLAT:.*]] = shufflevector <8 x i32> %[[INSERT]], <8 x i32> poison, <8 x i32> zeroinitializer
  // CHECK: %[[CONV:.*]] = icmp ne <8 x i32> %[[SPLAT]], zeroinitializer
  // CHECK: store <8 x i1> %[[CONV]], ptr %[[RETPTR:.*]]
}

vector8bool splat_uint_var_vector8bool(unsigned x) {
  return __builtin_splatvector(x, vector8bool);
  // CHECK-LABEL: @splat_uint_var_vector8bool
  // CHECK: store i32 %x, ptr %[[WHERE:.*]]
  // CHECK: %[[WHAT:.*]] = load i32, ptr %[[WHERE]]
  // CHECK: %[[INSERT:.*]] = insertelement <8 x i32> poison, i32 %[[WHAT]], i64 0
  // CHECK: %[[SPLAT:.*]] = shufflevector <8 x i32> %[[INSERT]], <8 x i32> poison, <8 x i32> zeroinitializer
  // CHECK: %[[CONV:.*]] = icmp ne <8 x i32> %[[SPLAT]], zeroinitializer
  // CHECK: store <8 x i1> %[[CONV]], ptr %[[RETPTR:.*]]
}

vector8bool splat_float_var_vector8bool(float x) {
  return __builtin_splatvector(x, vector8bool);
  // CHECK-LABEL: @splat_float_var_vector8bool
  // CHECK: store float %x, ptr %[[WHERE:.*]]
  // CHECK: %[[WHAT:.*]] = load float, ptr %[[WHERE]]
  // CHECK: %[[INSERT:.*]] = insertelement <8 x float> poison, float %[[WHAT]], i64 0
  // CHECK: %[[SPLAT:.*]] = shufflevector <8 x float> %[[INSERT]], <8 x float> poison, <8 x i32> zeroinitializer
  // CHECK: %[[CONV:.*]] = fcmp une <8 x float> %[[SPLAT]], zeroinitializer
  // CHECK: store <8 x i1> %[[CONV]], ptr %[[RETPTR:.*]]
}

vector8bool splat_double_var_vector8bool(double x) {
  return __builtin_splatvector(x, vector8bool);
  // CHECK-LABEL: @splat_double_var_vector8bool
  // CHECK: store double %x, ptr %[[WHERE:.*]]
  // CHECK: %[[WHAT:.*]] = load double, ptr %[[WHERE]]
  // CHECK: %[[INSERT:.*]] = insertelement <8 x double> poison, double %[[WHAT]], i64 0
  // CHECK: %[[SPLAT:.*]] = shufflevector <8 x double> %[[INSERT]], <8 x double> poison, <8 x i32> zeroinitializer
  // CHECK: %[[CONV:.*]] = fcmp une <8 x double> %[[SPLAT]], zeroinitializer
  // CHECK: store <8 x i1> %[[CONV]], ptr %[[RETPTR:.*]]
}

vector8long splat_bool_imm_vector8long(void) {
  return __builtin_splatvector(true, vector8long);
  // CHECK-LABEL: @splat_bool_imm_vector8long
  // CHECK: ret <8 x i64> splat (i64 1)
}

vector8long splat_int_imm_vector8long(void) {
  return __builtin_splatvector(-1, vector8long);
  // CHECK-LABEL: @splat_int_imm_vector8long
  // CHECK: ret <8 x i64> splat (i64 -1)
}

vector8long splat_uint_imm_vector8long(void) {
  return __builtin_splatvector(1U, vector8long);
  // CHECK-LABEL: @splat_uint_imm_vector8long
  // CHECK: ret <8 x i64> splat (i64 1)
}

vector8long splat_float_imm_vector8long(void) {
  return __builtin_splatvector(1.0f, vector8long);
  // CHECK-LABEL: @splat_float_imm_vector8long
  // CHECK: ret <8 x i64> splat (i64 1)
}

vector8long splat_double_imm_vector8long(void) {
  return __builtin_splatvector(1.0, vector8long);
  // CHECK-LABEL: @splat_double_imm_vector8long
  // CHECK: ret <8 x i64> splat (i64 1)
}

vector8long splat_bool_var_vector8long(bool x) {
  return __builtin_splatvector(x, vector8long);
  // CHECK-LABEL: @splat_bool_var_vector8long
  // CHECK: %[[STOREDV:.*]] = zext i1 %x to i8
  // CHECK: store i8 %[[STOREDV]], ptr %[[WHERE:.*]]
  // CHECK: %[[WHAT:.*]] = load i8, ptr %[[WHERE]]
  // CHECK: %[[WHAT_I1:.*]] = icmp ne i8 %[[WHAT]], 0
  // CHECK: %[[INSERT:.*]] = insertelement <8 x i1> poison, i1 %[[WHAT_I1]], i64 0
  // CHECK: %[[SPLAT:.*]] = shufflevector <8 x i1> %[[INSERT]], <8 x i1> poison, <8 x i32> zeroinitializer
  // CHECK: %[[CONV:.*]] = zext <8 x i1> %[[SPLAT]] to <8 x i64>
  // CHECK: ret <8 x i64> %[[CONV]]
}

vector8long splat_int_var_vector8long(int x) {
  return __builtin_splatvector(x, vector8long);
  // CHECK-LABEL: @splat_int_var_vector8long
  // CHECK: store i32 %x, ptr %[[WHERE:.*]]
  // CHECK: %[[WHAT:.*]] = load i32, ptr %[[WHERE]]
  // CHECK: %[[INSERT:.*]] = insertelement <8 x i32> poison, i32 %[[WHAT]], i64 0
  // CHECK: %[[SPLAT:.*]] = shufflevector <8 x i32> %[[INSERT]], <8 x i32> poison, <8 x i32> zeroinitializer
  // CHECK: %[[CONV:.*]] = sext <8 x i32> %[[SPLAT]] to <8 x i64>
  // CHECK: ret <8 x i64> %[[CONV]]
}

vector8long splat_uint_var_vector8long(unsigned x) {
  return __builtin_splatvector(x, vector8long);
  // CHECK-LABEL: @splat_uint_var_vector8long
  // CHECK: store i32 %x, ptr %[[WHERE:.*]]
  // CHECK: %[[WHAT:.*]] = load i32, ptr %[[WHERE]]
  // CHECK: %[[INSERT:.*]] = insertelement <8 x i32> poison, i32 %[[WHAT]], i64 0
  // CHECK: %[[SPLAT:.*]] = shufflevector <8 x i32> %[[INSERT]], <8 x i32> poison, <8 x i32> zeroinitializer
  // CHECK: %[[CONV:.*]] = zext <8 x i32> %[[SPLAT]] to <8 x i64>
  // CHECK: ret <8 x i64> %[[CONV]]
}

vector8long splat_float_var_vector8long(float x) {
  return __builtin_splatvector(x, vector8long);
  // CHECK-LABEL: @splat_float_var_vector8long
  // CHECK: store float %x, ptr %[[WHERE:.*]]
  // CHECK: %[[WHAT:.*]] = load float, ptr %[[WHERE]]
  // CHECK: %[[INSERT:.*]] = insertelement <8 x float> poison, float %[[WHAT]], i64 0
  // CHECK: %[[SPLAT:.*]] = shufflevector <8 x float> %[[INSERT]], <8 x float> poison, <8 x i32> zeroinitializer
  // CHECK: %[[CONV:.*]] = fptosi <8 x float> %[[SPLAT]] to <8 x i64>
  // CHECK: ret <8 x i64> %[[CONV]]
}

vector8long splat_double_var_vector8long(double x) {
  return __builtin_splatvector(x, vector8long);
  // CHECK-LABEL: @splat_double_var_vector8long
  // CHECK: store double %x, ptr %[[WHERE:.*]]
  // CHECK: %[[WHAT:.*]] = load double, ptr %[[WHERE]]
  // CHECK: %[[INSERT:.*]] = insertelement <8 x double> poison, double %[[WHAT]], i64 0
  // CHECK: %[[SPLAT:.*]] = shufflevector <8 x double> %[[INSERT]], <8 x double> poison, <8 x i32> zeroinitializer
  // CHECK: %[[CONV:.*]] = fptosi <8 x double> %[[SPLAT]] to <8 x i64>
  // CHECK: ret <8 x i64> %[[CONV]]
}

vector8ulong splat_bool_imm_vector8ulong(void) {
  return __builtin_splatvector(true, vector8ulong);
  // CHECK-LABEL: @splat_bool_imm_vector8ulong
  // CHECK: ret <8 x i64> splat (i64 1)
}

vector8ulong splat_int_imm_vector8ulong(void) {
  return __builtin_splatvector(-1, vector8ulong);
  // CHECK-LABEL: @splat_int_imm_vector8ulong
  // CHECK: ret <8 x i64> splat (i64 -1)
}

vector8ulong splat_uint_imm_vector8ulong(void) {
  return __builtin_splatvector(1U, vector8ulong);
  // CHECK-LABEL: @splat_uint_imm_vector8ulong
  // CHECK: ret <8 x i64> splat (i64 1)
}

vector8ulong splat_float_imm_vector8ulong(void) {
  return __builtin_splatvector(1.0f, vector8ulong);
  // CHECK-LABEL: @splat_float_imm_vector8ulong
  // CHECK: ret <8 x i64> splat (i64 1)
}

vector8ulong splat_double_imm_vector8ulong(void) {
  return __builtin_splatvector(1.0, vector8ulong);
  // CHECK-LABEL: @splat_double_imm_vector8ulong
  // CHECK: ret <8 x i64> splat (i64 1)
}

vector8ulong splat_bool_var_vector8ulong(bool x) {
  return __builtin_splatvector(x, vector8ulong);
  // CHECK-LABEL: @splat_bool_var_vector8ulong
  // CHECK: %[[STOREDV:.*]] = zext i1 %x to i8
  // CHECK: store i8 %[[STOREDV]], ptr %[[WHERE:.*]]
  // CHECK: %[[WHAT:.*]] = load i8, ptr %[[WHERE]]
  // CHECK: %[[WHAT_I1:.*]] = icmp ne i8 %[[WHAT]], 0
  // CHECK: %[[INSERT:.*]] = insertelement <8 x i1> poison, i1 %[[WHAT_I1]], i64 0
  // CHECK: %[[SPLAT:.*]] = shufflevector <8 x i1> %[[INSERT]], <8 x i1> poison, <8 x i32> zeroinitializer
  // CHECK: %[[CONV:.*]] = zext <8 x i1> %[[SPLAT]] to <8 x i64>
  // CHECK: ret <8 x i64> %[[CONV]]
}

vector8ulong splat_int_var_vector8ulong(int x) {
  return __builtin_splatvector(x, vector8ulong);
  // CHECK-LABEL: @splat_int_var_vector8ulong
  // CHECK: store i32 %x, ptr %[[WHERE:.*]]
  // CHECK: %[[WHAT:.*]] = load i32, ptr %[[WHERE]]
  // CHECK: %[[INSERT:.*]] = insertelement <8 x i32> poison, i32 %[[WHAT]], i64 0
  // CHECK: %[[SPLAT:.*]] = shufflevector <8 x i32> %[[INSERT]], <8 x i32> poison, <8 x i32> zeroinitializer
  // CHECK: %[[CONV:.*]] = sext <8 x i32> %[[SPLAT]] to <8 x i64>
  // CHECK: ret <8 x i64> %[[CONV]]
}

vector8ulong splat_uint_var_vector8ulong(unsigned x) {
  return __builtin_splatvector(x, vector8ulong);
  // CHECK-LABEL: @splat_uint_var_vector8ulong
  // CHECK: store i32 %x, ptr %[[WHERE:.*]]
  // CHECK: %[[WHAT:.*]] = load i32, ptr %[[WHERE]]
  // CHECK: %[[INSERT:.*]] = insertelement <8 x i32> poison, i32 %[[WHAT]], i64 0
  // CHECK: %[[SPLAT:.*]] = shufflevector <8 x i32> %[[INSERT]], <8 x i32> poison, <8 x i32> zeroinitializer
  // CHECK: %[[CONV:.*]] = zext <8 x i32> %[[SPLAT]] to <8 x i64>
  // CHECK: ret <8 x i64> %[[CONV]]
}

vector8ulong splat_float_var_vector8ulong(float x) {
  return __builtin_splatvector(x, vector8ulong);
  // CHECK-LABEL: @splat_float_var_vector8ulong
  // CHECK: store float %x, ptr %[[WHERE:.*]]
  // CHECK: %[[WHAT:.*]] = load float, ptr %[[WHERE]]
  // CHECK: %[[INSERT:.*]] = insertelement <8 x float> poison, float %[[WHAT]], i64 0
  // CHECK: %[[SPLAT:.*]] = shufflevector <8 x float> %[[INSERT]], <8 x float> poison, <8 x i32> zeroinitializer
  // CHECK: %[[CONV:.*]] = fptoui <8 x float> %[[SPLAT]] to <8 x i64>
  // CHECK: ret <8 x i64> %[[CONV]]
}

vector8ulong splat_double_var_vector8ulong(double x) {
  return __builtin_splatvector(x, vector8ulong);
  // CHECK-LABEL: @splat_double_var_vector8ulong
  // CHECK: store double %x, ptr %[[WHERE:.*]]
  // CHECK: %[[WHAT:.*]] = load double, ptr %[[WHERE]]
  // CHECK: %[[INSERT:.*]] = insertelement <8 x double> poison, double %[[WHAT]], i64 0
  // CHECK: %[[SPLAT:.*]] = shufflevector <8 x double> %[[INSERT]], <8 x double> poison, <8 x i32> zeroinitializer
  // CHECK: %[[CONV:.*]] = fptoui <8 x double> %[[SPLAT]] to <8 x i64>
  // CHECK: ret <8 x i64> %[[CONV]]
}

vector8float splat_bool_imm_vector8float(void) {
  return __builtin_splatvector(true, vector8float);
  // CHECK-LABEL: @splat_bool_imm_vector8float
  // CHECK: ret <8 x float> splat (float 1.000000e+00)
}

vector8float splat_int_imm_vector8float(void) {
  return __builtin_splatvector(-1, vector8float);
  // CHECK-LABEL: @splat_int_imm_vector8float
  // CHECK: ret <8 x float> splat (float -1.000000e+00)
}

vector8float splat_uint_imm_vector8float(void) {
  return __builtin_splatvector(1U, vector8float);
  // CHECK-LABEL: @splat_uint_imm_vector8float
  // CHECK: ret <8 x float> splat (float 1.000000e+00)
}

vector8float splat_float_imm_vector8float(void) {
  return __builtin_splatvector(1.0f, vector8float);
  // CHECK-LABEL: @splat_float_imm_vector8float
  // CHECK: ret <8 x float> splat (float 1.000000e+00)
}

vector8float splat_double_imm_vector8float(void) {
  return __builtin_splatvector(1.0, vector8float);
  // CHECK-LABEL: @splat_double_imm_vector8float
  // CHECK: ret <8 x float> splat (float 1.000000e+00)
}

vector8float splat_bool_var_vector8float(bool x) {
  return __builtin_splatvector(x, vector8float);
  // CHECK-LABEL: @splat_bool_var_vector8float
  // CHECK: %[[STOREDV:.*]] = zext i1 %x to i8
  // CHECK: store i8 %[[STOREDV]], ptr %[[WHERE:.*]]
  // CHECK: %[[WHAT:.*]] = load i8, ptr %[[WHERE]]
  // CHECK: %[[WHAT_I1:.*]] = icmp ne i8 %[[WHAT]], 0
  // CHECK: %[[INSERT:.*]] = insertelement <8 x i1> poison, i1 %[[WHAT_I1]], i64 0
  // CHECK: %[[SPLAT:.*]] = shufflevector <8 x i1> %[[INSERT]], <8 x i1> poison, <8 x i32> zeroinitializer
  // CHECK: %[[CONV:.*]] = uitofp <8 x i1> %[[SPLAT]] to <8 x float>
  // CHECK: ret <8 x float> %[[CONV]]
}

vector8float splat_int_var_vector8float(int x) {
  return __builtin_splatvector(x, vector8float);
  // CHECK-LABEL: @splat_int_var_vector8float
  // CHECK: store i32 %x, ptr %[[WHERE:.*]]
  // CHECK: %[[WHAT:.*]] = load i32, ptr %[[WHERE]]
  // CHECK: %[[INSERT:.*]] = insertelement <8 x i32> poison, i32 %[[WHAT]], i64 0
  // CHECK: %[[SPLAT:.*]] = shufflevector <8 x i32> %[[INSERT]], <8 x i32> poison, <8 x i32> zeroinitializer
  // CHECK: %[[CONV:.*]] = sitofp <8 x i32> %[[SPLAT]] to <8 x float>
  // CHECK: ret <8 x float> %[[CONV]]
}

vector8float splat_uint_var_vector8float(unsigned x) {
  return __builtin_splatvector(x, vector8float);
  // CHECK-LABEL: @splat_uint_var_vector8float
  // CHECK: store i32 %x, ptr %[[WHERE:.*]]
  // CHECK: %[[WHAT:.*]] = load i32, ptr %[[WHERE]]
  // CHECK: %[[INSERT:.*]] = insertelement <8 x i32> poison, i32 %[[WHAT]], i64 0
  // CHECK: %[[SPLAT:.*]] = shufflevector <8 x i32> %[[INSERT]], <8 x i32> poison, <8 x i32> zeroinitializer
  // CHECK: %[[CONV:.*]] = uitofp <8 x i32> %[[SPLAT]] to <8 x float>
  // CHECK: ret <8 x float> %[[CONV]]
}

vector8float splat_float_var_vector8float(float x) {
  return __builtin_splatvector(x, vector8float);
  // CHECK-LABEL: @splat_float_var_vector8float
  // CHECK: store float %x, ptr %[[WHERE:.*]]
  // CHECK: %[[WHAT:.*]] = load float, ptr %[[WHERE]]
  // CHECK: %[[INSERT:.*]] = insertelement <8 x float> poison, float %[[WHAT]], i64 0
  // CHECK: %[[SPLAT:.*]] = shufflevector <8 x float> %[[INSERT]], <8 x float> poison, <8 x i32> zeroinitializer
  // CHECK: ret <8 x float> %[[SPLAT]]
}

vector8float splat_double_var_vector8float(double x) {
  return __builtin_splatvector(x, vector8float);
  // CHECK-LABEL: @splat_double_var_vector8float
  // CHECK: store double %x, ptr %[[WHERE:.*]]
  // CHECK: %[[WHAT:.*]] = load double, ptr %[[WHERE]]
  // CHECK: %[[INSERT:.*]] = insertelement <8 x double> poison, double %[[WHAT]], i64 0
  // CHECK: %[[SPLAT:.*]] = shufflevector <8 x double> %[[INSERT]], <8 x double> poison, <8 x i32> zeroinitializer
  // CHECK: %[[CONV:.*]] = fptrunc <8 x double> %[[SPLAT]] to <8 x float>
  // CHECK: ret <8 x float> %[[CONV]]
}

vector8double splat_bool_imm_vector8double(void) {
  return __builtin_splatvector(true, vector8double);
  // CHECK-LABEL: @splat_bool_imm_vector8double
  // CHECK: ret <8 x double> splat (double 1.000000e+00)
}

vector8double splat_int_imm_vector8double(void) {
  return __builtin_splatvector(-1, vector8double);
  // CHECK-LABEL: @splat_int_imm_vector8double
  // CHECK: ret <8 x double> splat (double -1.000000e+00)
}

vector8double splat_uint_imm_vector8double(void) {
  return __builtin_splatvector(1U, vector8double);
  // CHECK-LABEL: @splat_uint_imm_vector8double
  // CHECK: ret <8 x double> splat (double 1.000000e+00)
}

vector8double splat_float_imm_vector8double(void) {
  return __builtin_splatvector(1.0f, vector8double);
  // CHECK-LABEL: @splat_float_imm_vector8double
  // CHECK: ret <8 x double> splat (double 1.000000e+00)
}

vector8double splat_double_imm_vector8double(void) {
  return __builtin_splatvector(1.0, vector8double);
  // CHECK-LABEL: @splat_double_imm_vector8double
  // CHECK: ret <8 x double> splat (double 1.000000e+00)
}

vector8double splat_bool_var_vector8double(bool x) {
  return __builtin_splatvector(x, vector8double);
  // CHECK-LABEL: @splat_bool_var_vector8double
  // CHECK: %[[STOREDV:.*]] = zext i1 %x to i8
  // CHECK: store i8 %[[STOREDV]], ptr %[[WHERE:.*]]
  // CHECK: %[[WHAT:.*]] = load i8, ptr %[[WHERE]]
  // CHECK: %[[WHAT_I1:.*]] = icmp ne i8 %[[WHAT]], 0
  // CHECK: %[[INSERT:.*]] = insertelement <8 x i1> poison, i1 %[[WHAT_I1]], i64 0
  // CHECK: %[[SPLAT:.*]] = shufflevector <8 x i1> %[[INSERT]], <8 x i1> poison, <8 x i32> zeroinitializer
  // CHECK: %[[CONV:.*]] = uitofp <8 x i1> %[[SPLAT]] to <8 x double>
  // CHECK: ret <8 x double> %[[CONV]]
}

vector8double splat_int_var_vector8double(int x) {
  return __builtin_splatvector(x, vector8double);
  // CHECK-LABEL: @splat_int_var_vector8double
  // CHECK: store i32 %x, ptr %[[WHERE:.*]]
  // CHECK: %[[WHAT:.*]] = load i32, ptr %[[WHERE]]
  // CHECK: %[[INSERT:.*]] = insertelement <8 x i32> poison, i32 %[[WHAT]], i64 0
  // CHECK: %[[SPLAT:.*]] = shufflevector <8 x i32> %[[INSERT]], <8 x i32> poison, <8 x i32> zeroinitializer
  // CHECK: %[[CONV:.*]] = sitofp <8 x i32> %[[SPLAT]] to <8 x double>
  // CHECK: ret <8 x double> %[[CONV]]
}

vector8double splat_uint_var_vector8double(unsigned x) {
  return __builtin_splatvector(x, vector8double);
  // CHECK-LABEL: @splat_uint_var_vector8double
  // CHECK: store i32 %x, ptr %[[WHERE:.*]]
  // CHECK: %[[WHAT:.*]] = load i32, ptr %[[WHERE]]
  // CHECK: %[[INSERT:.*]] = insertelement <8 x i32> poison, i32 %[[WHAT]], i64 0
  // CHECK: %[[SPLAT:.*]] = shufflevector <8 x i32> %[[INSERT]], <8 x i32> poison, <8 x i32> zeroinitializer
  // CHECK: %[[CONV:.*]] = uitofp <8 x i32> %[[SPLAT]] to <8 x double>
  // CHECK: ret <8 x double> %[[CONV]]
}

vector8double splat_float_var_vector8double(float x) {
  return __builtin_splatvector(x, vector8double);
  // CHECK-LABEL: @splat_float_var_vector8double
  // CHECK: store float %x, ptr %[[WHERE:.*]]
  // CHECK: %[[WHAT:.*]] = load float, ptr %[[WHERE]]
  // CHECK: %[[INSERT:.*]] = insertelement <8 x float> poison, float %[[WHAT]], i64 0
  // CHECK: %[[SPLAT:.*]] = shufflevector <8 x float> %[[INSERT]], <8 x float> poison, <8 x i32> zeroinitializer
  // CHECK: %[[CONV:.*]] = fpext <8 x float> %[[SPLAT]] to <8 x double>
  // CHECK: ret <8 x double> %[[CONV]]
}

vector8double splat_double_var_vector8double(double x) {
  return __builtin_splatvector(x, vector8double);
  // CHECK-LABEL: @splat_double_var_vector8double
  // CHECK: store double %x, ptr %[[WHERE:.*]]
  // CHECK: %[[WHAT:.*]] = load double, ptr %[[WHERE]]
  // CHECK: %[[INSERT:.*]] = insertelement <8 x double> poison, double %[[WHAT]], i64 0
  // CHECK: %[[SPLAT:.*]] = shufflevector <8 x double> %[[INSERT]], <8 x double> poison, <8 x i32> zeroinitializer
  // CHECK: ret <8 x double> %[[SPLAT]]
}

vector8float splat_flt_trunc(double x) {
  return __builtin_splatvector(x, vector8float);
  // CHECK-LABEL: @splat_flt_trunc
  // CHECK: fptrunc <8 x double> %{{.*}} to <8 x float>
}

vector8double splat_flt_ext(float x) {
  return __builtin_splatvector(x, vector8double);
  // CHECK-LABEL: @splat_flt_ext
  // CHECK: fpext <8 x float> %{{.*}} to <8 x double>
}

vector8long splat_flt_tosi(float x) {
  return __builtin_splatvector(x, vector8long);
  // CHECK-LABEL: @splat_flt_tosi
  // CHECK: fptosi <8 x float> %{{.*}} to <8 x i64>
}

vector8ulong splat_flt_toui(float x) {
  return __builtin_splatvector(x, vector8ulong);
  // CHECK-LABEL: @splat_flt_toui
  // CHECK: fptoui <8 x float> %{{.*}} to <8 x i64>
}

vector8ulong splat_fltd_toui(double x) {
  return __builtin_splatvector(x, vector8ulong);
  // CHECK-LABEL: @splat_fltd_toui
  // CHECK: fptoui <8 x double> %{{.*}} to <8 x i64>
}

vector8ulong splat_int_zext(unsigned short x) {
  return __builtin_splatvector(x, vector8ulong);
  // CHECK-LABEL: @splat_int_zext
  // CHECK: zext <8 x i16> %{{.*}} to <8 x i64>
}

vector8long splat_int_sext(short x) {
  return __builtin_splatvector(x, vector8long);
  // CHECK-LABEL: @splat_int_sext
  // CHECK: sext <8 x i16> %{{.*}} to <8 x i64>
}

vector8float splat_int_tofp(short x) {
  return __builtin_splatvector(x, vector8float);
  // CHECK-LABEL: @splat_int_tofp
  // CHECK: sitofp <8 x i16> %{{.*}} to <8 x float>
}

vector8float splat_uint_tofp(unsigned short x) {
  return __builtin_splatvector(x, vector8float);
  // CHECK-LABEL: @splat_uint_tofp
  // CHECK: uitofp <8 x i16> %{{.*}} to <8 x float>
}

#ifdef __cplusplus
}
#endif


#ifdef __cplusplus
template<typename T>
T splat_int_toT(long x) {
  return __builtin_splatvector(x, T);
}

extern "C" {
  vector8double splat_int_toT_fp(long x) {
    // CHECK-LABEL: @splat_int_toT_fp
    // CHECK: sitofp <8 x i64> %{{.*}} to <8 x double>
    return splat_int_toT<vector8double>(x);
  }
}
#else
vector8double splat_int_toT_fp(long x) {
  return __builtin_splatvector(x, vector8double);
}
#endif
