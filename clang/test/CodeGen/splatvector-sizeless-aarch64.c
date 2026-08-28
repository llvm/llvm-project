// RUN: %clang_cc1 -triple aarch64 -target-feature +sve -emit-llvm %s -o - | FileCheck %s
// RUN: %clang_cc1 -triple aarch64 -target-feature +sve -emit-llvm -x c++ %s -o - | FileCheck %s

#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

__SVBool_t splat_bool_imm___SVBool_t(void) {
  return __builtin_splatvector(true, __SVBool_t);
  // CHECK-LABEL: @splat_bool_imm___SVBool_t
  // CHECK: ret <vscale x {{[0-9]*}} x i1> splat (i1 true)
}

__SVBool_t splat_int_imm___SVBool_t(void) {
  return __builtin_splatvector(-1, __SVBool_t);
  // CHECK-LABEL: @splat_int_imm___SVBool_t
  // CHECK: ret <vscale x {{[0-9]*}} x i1> splat (i1 true)
}

__SVBool_t splat_uint_imm___SVBool_t(void) {
  return __builtin_splatvector(1U, __SVBool_t);
  // CHECK-LABEL: @splat_uint_imm___SVBool_t
  // CHECK: ret <vscale x {{[0-9]*}} x i1> splat (i1 true)
}

__SVBool_t splat_float_imm___SVBool_t(void) {
  return __builtin_splatvector(1.0f, __SVBool_t);
  // CHECK-LABEL: @splat_float_imm___SVBool_t
  // CHECK: ret <vscale x {{[0-9]*}} x i1> splat (i1 true)
}

__SVBool_t splat_double_imm___SVBool_t(void) {
  return __builtin_splatvector(1.0, __SVBool_t);
  // CHECK-LABEL: @splat_double_imm___SVBool_t
  // CHECK: ret <vscale x {{[0-9]*}} x i1> splat (i1 true)
}

__SVBool_t splat_bool_var___SVBool_t(bool x) {
  return __builtin_splatvector(x, __SVBool_t);
  // CHECK-LABEL: @splat_bool_var___SVBool_t
  // CHECK: %[[STOREDV:.*]] = zext i1 %x to i8
  // CHECK: store i8 %[[STOREDV]], ptr %[[WHERE:.*]]
  // CHECK: %[[WHAT:.*]] = load i8, ptr %[[WHERE]]
  // CHECK: %[[WHAT_I1:.*]] = icmp ne i8 %[[WHAT]], 0
  // CHECK: %[[INSERT:.*]] = insertelement <vscale x {{[0-9]*}} x i1> poison, i1 %[[WHAT_I1]], i64 0
  // CHECK: %[[SPLAT:.*]] = shufflevector <vscale x {{[0-9]*}} x i1> %[[INSERT]], <vscale x {{[0-9]*}} x i1> poison, <vscale x {{[0-9]*}} x i32> zeroinitializer
  // CHECK: ret <vscale x {{[0-9]*}} x i1> %[[SPLAT]]
}

__SVBool_t splat_int_var___SVBool_t(int x) {
  return __builtin_splatvector(x, __SVBool_t);
  // CHECK-LABEL: @splat_int_var___SVBool_t
  // CHECK: store i32 %x, ptr %[[WHERE:.*]]
  // CHECK: %[[WHAT:.*]] = load i32, ptr %[[WHERE]]
  // CHECK: %[[INSERT:.*]] = insertelement <vscale x {{[0-9]*}} x i32> poison, i32 %[[WHAT]], i64 0
  // CHECK: %[[SPLAT:.*]] = shufflevector <vscale x {{[0-9]*}} x i32> %[[INSERT]], <vscale x {{[0-9]*}} x i32> poison, <vscale x {{[0-9]*}} x i32> zeroinitializer
  // CHECK: %[[CONV:.*]] = icmp ne <vscale x {{[0-9]*}} x i32> %splatvector.splat, zeroinitializer
  // CHECK: ret <vscale x {{[0-9]*}} x i1> %[[CONV]]
}

__SVBool_t splat_uint_var___SVBool_t(unsigned x) {
  return __builtin_splatvector(x, __SVBool_t);
  // CHECK-LABEL: @splat_uint_var___SVBool_t
  // CHECK: store i32 %x, ptr %[[WHERE:.*]]
  // CHECK: %[[WHAT:.*]] = load i32, ptr %[[WHERE]]
  // CHECK: %[[INSERT:.*]] = insertelement <vscale x {{[0-9]*}} x i32> poison, i32 %[[WHAT]], i64 0
  // CHECK: %[[SPLAT:.*]] = shufflevector <vscale x {{[0-9]*}} x i32> %[[INSERT]], <vscale x {{[0-9]*}} x i32> poison, <vscale x {{[0-9]*}} x i32> zeroinitializer
  // CHECK: %[[CONV:.*]] = icmp ne <vscale x {{[0-9]*}} x i32> %[[SPLAT]], zeroinitializer
  // CHECK: ret <vscale x {{[0-9]*}} x i1> %[[CONV]]
}

__SVBool_t splat_float_var___SVBool_t(float x) {
  return __builtin_splatvector(x, __SVBool_t);
  // CHECK-LABEL: @splat_float_var___SVBool_t
  // CHECK: store float %x, ptr %[[WHERE:.*]]
  // CHECK: %[[WHAT:.*]] = load float, ptr %[[WHERE]]
  // CHECK: %[[INSERT:.*]] = insertelement <vscale x {{[0-9]*}} x float> poison, float %[[WHAT]], i64 0
  // CHECK: %[[SPLAT:.*]] = shufflevector <vscale x {{[0-9]*}} x float> %[[INSERT]], <vscale x {{[0-9]*}} x float> poison, <vscale x {{[0-9]*}} x i32> zeroinitializer
  // CHECK: %[[CONV:.*]] = fcmp une <vscale x {{[0-9]*}} x float> %[[SPLAT]], zeroinitializer
  // CHECK: ret <vscale x {{[0-9]*}} x i1> %[[CONV]]
}

__SVBool_t splat_double_var___SVBool_t(double x) {
  return __builtin_splatvector(x, __SVBool_t);
  // CHECK-LABEL: @splat_double_var___SVBool_t
  // CHECK: store double %x, ptr %[[WHERE:.*]]
  // CHECK: %[[WHAT:.*]] = load double, ptr %[[WHERE]]
  // CHECK: %[[INSERT:.*]] = insertelement <vscale x {{[0-9]*}} x double> poison, double %[[WHAT]], i64 0
  // CHECK: %[[SPLAT:.*]] = shufflevector <vscale x {{[0-9]*}} x double> %[[INSERT]], <vscale x {{[0-9]*}} x double> poison, <vscale x {{[0-9]*}} x i32> zeroinitializer
  // CHECK: %[[CONV:.*]] = fcmp une <vscale x {{[0-9]*}} x double> %[[SPLAT]], zeroinitializer
  // CHECK: ret <vscale x {{[0-9]*}} x i1> %[[CONV]]
}

__SVInt64_t splat_bool_imm___SVInt64_t(void) {
  return __builtin_splatvector(true, __SVInt64_t);
  // CHECK-LABEL: @splat_bool_imm___SVInt64_t
  // CHECK: ret <vscale x {{[0-9]*}} x i64> splat (i64 1)
}

__SVInt64_t splat_int_imm___SVInt64_t(void) {
  return __builtin_splatvector(-1, __SVInt64_t);
  // CHECK-LABEL: @splat_int_imm___SVInt64_t
  // CHECK: ret <vscale x {{[0-9]*}} x i64> splat (i64 -1)
}

__SVInt64_t splat_uint_imm___SVInt64_t(void) {
  return __builtin_splatvector(1U, __SVInt64_t);
  // CHECK-LABEL: @splat_uint_imm___SVInt64_t
  // CHECK: ret <vscale x {{[0-9]*}} x i64> splat (i64 1)
}

__SVInt64_t splat_float_imm___SVInt64_t(void) {
  return __builtin_splatvector(1.0f, __SVInt64_t);
  // CHECK-LABEL: @splat_float_imm___SVInt64_t
  // CHECK: ret <vscale x {{[0-9]*}} x i64> splat (i64 1)
}

__SVInt64_t splat_double_imm___SVInt64_t(void) {
  return __builtin_splatvector(1.0, __SVInt64_t);
  // CHECK-LABEL: @splat_double_imm___SVInt64_t
  // CHECK: ret <vscale x {{[0-9]*}} x i64> splat (i64 1)
}

__SVInt64_t splat_bool_var___SVInt64_t(bool x) {
  return __builtin_splatvector(x, __SVInt64_t);
  // CHECK-LABEL: @splat_bool_var___SVInt64_t
  // CHECK: %[[STOREDV:.*]] = zext i1 %x to i8
  // CHECK: store i8 %[[STOREDV]], ptr %[[WHERE:.*]]
  // CHECK: %[[WHAT:.*]] = load i8, ptr %[[WHERE]]
  // CHECK: %[[WHAT_I1:.*]] = icmp ne i8 %[[WHAT]], 0
  // CHECK: %[[INSERT:.*]] = insertelement <vscale x {{[0-9]*}} x i1> poison, i1 %[[WHAT_I1]], i64 0
  // CHECK: %[[SPLAT:.*]] = shufflevector <vscale x {{[0-9]*}} x i1> %[[INSERT]], <vscale x {{[0-9]*}} x i1> poison, <vscale x {{[0-9]*}} x i32> zeroinitializer
  // CHECK: %[[CONV:.*]] = zext <vscale x {{[0-9]*}} x i1> %[[SPLAT]] to <vscale x {{[0-9]*}} x i64>
  // CHECK: ret <vscale x {{[0-9]*}} x i64> %[[CONV]]
}

__SVInt64_t splat_int_var___SVInt64_t(int x) {
  return __builtin_splatvector(x, __SVInt64_t);
  // CHECK-LABEL: @splat_int_var___SVInt64_t
  // CHECK: store i32 %x, ptr %[[WHERE:.*]]
  // CHECK: %[[WHAT:.*]] = load i32, ptr %[[WHERE]]
  // CHECK: %[[INSERT:.*]] = insertelement <vscale x {{[0-9]*}} x i32> poison, i32 %[[WHAT]], i64 0
  // CHECK: %[[SPLAT:.*]] = shufflevector <vscale x {{[0-9]*}} x i32> %[[INSERT]], <vscale x {{[0-9]*}} x i32> poison, <vscale x {{[0-9]*}} x i32> zeroinitializer
  // CHECK: %[[CONV:.*]] = sext <vscale x {{[0-9]*}} x i32> %[[SPLAT]] to <vscale x {{[0-9]*}} x i64>
  // CHECK: ret <vscale x {{[0-9]*}} x i64> %[[CONV]]
}

__SVInt64_t splat_uint_var___SVInt64_t(unsigned x) {
  return __builtin_splatvector(x, __SVInt64_t);
  // CHECK-LABEL: @splat_uint_var___SVInt64_t
  // CHECK: store i32 %x, ptr %[[WHERE:.*]]
  // CHECK: %[[WHAT:.*]] = load i32, ptr %[[WHERE]]
  // CHECK: %[[INSERT:.*]] = insertelement <vscale x {{[0-9]*}} x i32> poison, i32 %[[WHAT]], i64 0
  // CHECK: %[[SPLAT:.*]] = shufflevector <vscale x {{[0-9]*}} x i32> %[[INSERT]], <vscale x {{[0-9]*}} x i32> poison, <vscale x {{[0-9]*}} x i32> zeroinitializer
  // CHECK: %[[CONV:.*]] = zext <vscale x {{[0-9]*}} x i32> %[[SPLAT]] to <vscale x {{[0-9]*}} x i64>
  // CHECK: ret <vscale x {{[0-9]*}} x i64> %[[CONV]]
}

__SVInt64_t splat_float_var___SVInt64_t(float x) {
  return __builtin_splatvector(x, __SVInt64_t);
  // CHECK-LABEL: @splat_float_var___SVInt64_t
  // CHECK: store float %x, ptr %[[WHERE:.*]]
  // CHECK: %[[WHAT:.*]] = load float, ptr %[[WHERE]]
  // CHECK: %[[INSERT:.*]] = insertelement <vscale x {{[0-9]*}} x float> poison, float %[[WHAT]], i64 0
  // CHECK: %[[SPLAT:.*]] = shufflevector <vscale x {{[0-9]*}} x float> %[[INSERT]], <vscale x {{[0-9]*}} x float> poison, <vscale x {{[0-9]*}} x i32> zeroinitializer
  // CHECK: %[[CONV:.*]] = fptosi <vscale x {{[0-9]*}} x float> %[[SPLAT]] to <vscale x {{[0-9]*}} x i64>
  // CHECK: ret <vscale x {{[0-9]*}} x i64> %[[CONV]]
}

__SVInt64_t splat_double_var___SVInt64_t(double x) {
  return __builtin_splatvector(x, __SVInt64_t);
  // CHECK-LABEL: @splat_double_var___SVInt64_t
  // CHECK: store double %x, ptr %[[WHERE:.*]]
  // CHECK: %[[WHAT:.*]] = load double, ptr %[[WHERE]]
  // CHECK: %[[INSERT:.*]] = insertelement <vscale x {{[0-9]*}} x double> poison, double %[[WHAT]], i64 0
  // CHECK: %[[SPLAT:.*]] = shufflevector <vscale x {{[0-9]*}} x double> %[[INSERT]], <vscale x {{[0-9]*}} x double> poison, <vscale x {{[0-9]*}} x i32> zeroinitializer
  // CHECK: %[[CONV:.*]] = fptosi <vscale x {{[0-9]*}} x double> %[[SPLAT]] to <vscale x {{[0-9]*}} x i64>
  // CHECK: ret <vscale x {{[0-9]*}} x i64> %[[CONV]]
}

__SVUint64_t splat_bool_imm___SVUint64_t(void) {
  return __builtin_splatvector(true, __SVUint64_t);
  // CHECK-LABEL: @splat_bool_imm___SVUint64_t
  // CHECK: ret <vscale x {{[0-9]*}} x i64> splat (i64 1)
}

__SVUint64_t splat_int_imm___SVUint64_t(void) {
  return __builtin_splatvector(-1, __SVUint64_t);
  // CHECK-LABEL: @splat_int_imm___SVUint64_t
  // CHECK: ret <vscale x {{[0-9]*}} x i64> splat (i64 -1)
}

__SVUint64_t splat_uint_imm___SVUint64_t(void) {
  return __builtin_splatvector(1U, __SVUint64_t);
  // CHECK-LABEL: @splat_uint_imm___SVUint64_t
  // CHECK: ret <vscale x {{[0-9]*}} x i64> splat (i64 1)
}

__SVUint64_t splat_float_imm___SVUint64_t(void) {
  return __builtin_splatvector(1.0f, __SVUint64_t);
  // CHECK-LABEL: @splat_float_imm___SVUint64_t
  // CHECK: ret <vscale x {{[0-9]*}} x i64> splat (i64 1)
}

__SVUint64_t splat_double_imm___SVUint64_t(void) {
  return __builtin_splatvector(1.0, __SVUint64_t);
  // CHECK-LABEL: @splat_double_imm___SVUint64_t
  // CHECK: ret <vscale x {{[0-9]*}} x i64> splat (i64 1)
}

__SVUint64_t splat_bool_var___SVUint64_t(bool x) {
  return __builtin_splatvector(x, __SVUint64_t);
  // CHECK-LABEL: @splat_bool_var___SVUint64_t
  // CHECK: %[[STOREDV:.*]] = zext i1 %x to i8
  // CHECK: store i8 %[[STOREDV]], ptr %[[WHERE:.*]]
  // CHECK: %[[WHAT:.*]] = load i8, ptr %[[WHERE]]
  // CHECK: %[[WHAT_I1:.*]] = icmp ne i8 %[[WHAT]], 0
  // CHECK: %[[INSERT:.*]] = insertelement <vscale x {{[0-9]*}} x i1> poison, i1 %[[WHAT_I1]], i64 0
  // CHECK: %[[SPLAT:.*]] = shufflevector <vscale x {{[0-9]*}} x i1> %[[INSERT]], <vscale x {{[0-9]*}} x i1> poison, <vscale x {{[0-9]*}} x i32> zeroinitializer
  // CHECK: %[[CONV:.*]] = zext <vscale x {{[0-9]*}} x i1> %[[SPLAT]] to <vscale x {{[0-9]*}} x i64>
  // CHECK: ret <vscale x {{[0-9]*}} x i64> %[[CONV]]
}

__SVUint64_t splat_int_var___SVUint64_t(int x) {
  return __builtin_splatvector(x, __SVUint64_t);
  // CHECK-LABEL: @splat_int_var___SVUint64_t
  // CHECK: store i32 %x, ptr %[[WHERE:.*]]
  // CHECK: %[[WHAT:.*]] = load i32, ptr %[[WHERE]]
  // CHECK: %[[INSERT:.*]] = insertelement <vscale x {{[0-9]*}} x i32> poison, i32 %[[WHAT]], i64 0
  // CHECK: %[[SPLAT:.*]] = shufflevector <vscale x {{[0-9]*}} x i32> %[[INSERT]], <vscale x {{[0-9]*}} x i32> poison, <vscale x {{[0-9]*}} x i32> zeroinitializer
  // CHECK: %[[CONV:.*]] = sext <vscale x {{[0-9]*}} x i32> %[[SPLAT]] to <vscale x {{[0-9]*}} x i64>
  // CHECK: ret <vscale x {{[0-9]*}} x i64> %[[CONV]]
}

__SVUint64_t splat_uint_var___SVUint64_t(unsigned x) {
  return __builtin_splatvector(x, __SVUint64_t);
  // CHECK-LABEL: @splat_uint_var___SVUint64_t
  // CHECK: store i32 %x, ptr %[[WHERE:.*]]
  // CHECK: %[[WHAT:.*]] = load i32, ptr %[[WHERE]]
  // CHECK: %[[INSERT:.*]] = insertelement <vscale x {{[0-9]*}} x i32> poison, i32 %[[WHAT]], i64 0
  // CHECK: %[[SPLAT:.*]] = shufflevector <vscale x {{[0-9]*}} x i32> %[[INSERT]], <vscale x {{[0-9]*}} x i32> poison, <vscale x {{[0-9]*}} x i32> zeroinitializer
  // CHECK: %[[CONV:.*]] = zext <vscale x {{[0-9]*}} x i32> %[[SPLAT]] to <vscale x {{[0-9]*}} x i64>
  // CHECK: ret <vscale x {{[0-9]*}} x i64> %[[CONV]]
}

__SVUint64_t splat_float_var___SVUint64_t(float x) {
  return __builtin_splatvector(x, __SVUint64_t);
  // CHECK-LABEL: @splat_float_var___SVUint64_t
  // CHECK: store float %x, ptr %[[WHERE:.*]]
  // CHECK: %[[WHAT:.*]] = load float, ptr %[[WHERE]]
  // CHECK: %[[INSERT:.*]] = insertelement <vscale x {{[0-9]*}} x float> poison, float %[[WHAT]], i64 0
  // CHECK: %[[SPLAT:.*]] = shufflevector <vscale x {{[0-9]*}} x float> %[[INSERT]], <vscale x {{[0-9]*}} x float> poison, <vscale x {{[0-9]*}} x i32> zeroinitializer
  // CHECK: %[[CONV:.*]] = fptoui <vscale x {{[0-9]*}} x float> %[[SPLAT]] to <vscale x {{[0-9]*}} x i64>
  // CHECK: ret <vscale x {{[0-9]*}} x i64> %[[CONV]]
}

__SVUint64_t splat_double_var___SVUint64_t(double x) {
  return __builtin_splatvector(x, __SVUint64_t);
  // CHECK-LABEL: @splat_double_var___SVUint64_t
  // CHECK: store double %x, ptr %[[WHERE:.*]]
  // CHECK: %[[WHAT:.*]] = load double, ptr %[[WHERE]]
  // CHECK: %[[INSERT:.*]] = insertelement <vscale x {{[0-9]*}} x double> poison, double %[[WHAT]], i64 0
  // CHECK: %[[SPLAT:.*]] = shufflevector <vscale x {{[0-9]*}} x double> %[[INSERT]], <vscale x {{[0-9]*}} x double> poison, <vscale x {{[0-9]*}} x i32> zeroinitializer
  // CHECK: %[[CONV:.*]] = fptoui <vscale x {{[0-9]*}} x double> %[[SPLAT]] to <vscale x {{[0-9]*}} x i64>
  // CHECK: ret <vscale x {{[0-9]*}} x i64> %[[CONV]]
}

__SVFloat32_t splat_bool_imm___SVFloat32_t(void) {
  return __builtin_splatvector(true, __SVFloat32_t);
  // CHECK-LABEL: @splat_bool_imm___SVFloat32_t
  // CHECK: ret <vscale x {{[0-9]*}} x float> splat (float 1.000000e+00)
}

__SVFloat32_t splat_int_imm___SVFloat32_t(void) {
  return __builtin_splatvector(-1, __SVFloat32_t);
  // CHECK-LABEL: @splat_int_imm___SVFloat32_t
  // CHECK: ret <vscale x {{[0-9]*}} x float> splat (float -1.000000e+00)
}

__SVFloat32_t splat_uint_imm___SVFloat32_t(void) {
  return __builtin_splatvector(1U, __SVFloat32_t);
  // CHECK-LABEL: @splat_uint_imm___SVFloat32_t
  // CHECK: ret <vscale x {{[0-9]*}} x float> splat (float 1.000000e+00)
}

__SVFloat32_t splat_float_imm___SVFloat32_t(void) {
  return __builtin_splatvector(1.0f, __SVFloat32_t);
  // CHECK-LABEL: @splat_float_imm___SVFloat32_t
  // CHECK: ret <vscale x {{[0-9]*}} x float> splat (float 1.000000e+00)
}

__SVFloat32_t splat_double_imm___SVFloat32_t(void) {
  return __builtin_splatvector(1.0, __SVFloat32_t);
  // CHECK-LABEL: @splat_double_imm___SVFloat32_t
  // CHECK: ret <vscale x {{[0-9]*}} x float> splat (float 1.000000e+00)
}

__SVFloat32_t splat_bool_var___SVFloat32_t(bool x) {
  return __builtin_splatvector(x, __SVFloat32_t);
  // CHECK-LABEL: @splat_bool_var___SVFloat32_t
  // CHECK: %[[STOREDV:.*]] = zext i1 %x to i8
  // CHECK: store i8 %[[STOREDV]], ptr %[[WHERE:.*]]
  // CHECK: %[[WHAT:.*]] = load i8, ptr %[[WHERE]]
  // CHECK: %[[WHAT_I1:.*]] = icmp ne i8 %[[WHAT]], 0
  // CHECK: %[[INSERT:.*]] = insertelement <vscale x {{[0-9]*}} x i1> poison, i1 %[[WHAT_I1]], i64 0
  // CHECK: %[[SPLAT:.*]] = shufflevector <vscale x {{[0-9]*}} x i1> %[[INSERT]], <vscale x {{[0-9]*}} x i1> poison, <vscale x {{[0-9]*}} x i32> zeroinitializer
  // CHECK: %[[CONV:.*]] = uitofp <vscale x {{[0-9]*}} x i1> %[[SPLAT]] to <vscale x {{[0-9]*}} x float>
  // CHECK: ret <vscale x {{[0-9]*}} x float> %[[CONV]]
}

__SVFloat32_t splat_int_var___SVFloat32_t(int x) {
  return __builtin_splatvector(x, __SVFloat32_t);
  // CHECK-LABEL: @splat_int_var___SVFloat32_t
  // CHECK: store i32 %x, ptr %[[WHERE:.*]]
  // CHECK: %[[WHAT:.*]] = load i32, ptr %[[WHERE]]
  // CHECK: %[[INSERT:.*]] = insertelement <vscale x {{[0-9]*}} x i32> poison, i32 %[[WHAT]], i64 0
  // CHECK: %[[SPLAT:.*]] = shufflevector <vscale x {{[0-9]*}} x i32> %[[INSERT]], <vscale x {{[0-9]*}} x i32> poison, <vscale x {{[0-9]*}} x i32> zeroinitializer
  // CHECK: %[[CONV:.*]] = sitofp <vscale x {{[0-9]*}} x i32> %[[SPLAT]] to <vscale x {{[0-9]*}} x float>
  // CHECK: ret <vscale x {{[0-9]*}} x float> %[[CONV]]
}

__SVFloat32_t splat_uint_var___SVFloat32_t(unsigned x) {
  return __builtin_splatvector(x, __SVFloat32_t);
  // CHECK-LABEL: @splat_uint_var___SVFloat32_t
  // CHECK: store i32 %x, ptr %[[WHERE:.*]]
  // CHECK: %[[WHAT:.*]] = load i32, ptr %[[WHERE]]
  // CHECK: %[[INSERT:.*]] = insertelement <vscale x {{[0-9]*}} x i32> poison, i32 %[[WHAT]], i64 0
  // CHECK: %[[SPLAT:.*]] = shufflevector <vscale x {{[0-9]*}} x i32> %[[INSERT]], <vscale x {{[0-9]*}} x i32> poison, <vscale x {{[0-9]*}} x i32> zeroinitializer
  // CHECK: %[[CONV:.*]] = uitofp <vscale x {{[0-9]*}} x i32> %[[SPLAT]] to <vscale x {{[0-9]*}} x float>
  // CHECK: ret <vscale x {{[0-9]*}} x float> %[[CONV]]
}

__SVFloat32_t splat_float_var___SVFloat32_t(float x) {
  return __builtin_splatvector(x, __SVFloat32_t);
  // CHECK-LABEL: @splat_float_var___SVFloat32_t
  // CHECK: store float %x, ptr %[[WHERE:.*]]
  // CHECK: %[[WHAT:.*]] = load float, ptr %[[WHERE]]
  // CHECK: %[[INSERT:.*]] = insertelement <vscale x {{[0-9]*}} x float> poison, float %[[WHAT]], i64 0
  // CHECK: %[[SPLAT:.*]] = shufflevector <vscale x {{[0-9]*}} x float> %[[INSERT]], <vscale x {{[0-9]*}} x float> poison, <vscale x {{[0-9]*}} x i32> zeroinitializer
  // CHECK: ret <vscale x {{[0-9]*}} x float> %[[SPLAT]]
}

__SVFloat32_t splat_double_var___SVFloat32_t(double x) {
  return __builtin_splatvector(x, __SVFloat32_t);
  // CHECK-LABEL: @splat_double_var___SVFloat32_t
  // CHECK: store double %x, ptr %[[WHERE:.*]]
  // CHECK: %[[WHAT:.*]] = load double, ptr %[[WHERE]]
  // CHECK: %[[INSERT:.*]] = insertelement <vscale x {{[0-9]*}} x double> poison, double %[[WHAT]], i64 0
  // CHECK: %[[SPLAT:.*]] = shufflevector <vscale x {{[0-9]*}} x double> %[[INSERT]], <vscale x {{[0-9]*}} x double> poison, <vscale x {{[0-9]*}} x i32> zeroinitializer
  // CHECK: %[[CONV:.*]] = fptrunc <vscale x {{[0-9]*}} x double> %[[SPLAT]] to <vscale x {{[0-9]*}} x float>
  // CHECK: ret <vscale x {{[0-9]*}} x float> %[[CONV]]
}

__SVFloat64_t splat_bool_imm___SVFloat64_t(void) {
  return __builtin_splatvector(true, __SVFloat64_t);
  // CHECK-LABEL: @splat_bool_imm___SVFloat64_t
  // CHECK: ret <vscale x {{[0-9]*}} x double> splat (double 1.000000e+00)
}

__SVFloat64_t splat_int_imm___SVFloat64_t(void) {
  return __builtin_splatvector(-1, __SVFloat64_t);
  // CHECK-LABEL: @splat_int_imm___SVFloat64_t
  // CHECK: ret <vscale x {{[0-9]*}} x double> splat (double -1.000000e+00)
}

__SVFloat64_t splat_uint_imm___SVFloat64_t(void) {
  return __builtin_splatvector(1U, __SVFloat64_t);
  // CHECK-LABEL: @splat_uint_imm___SVFloat64_t
  // CHECK: ret <vscale x {{[0-9]*}} x double> splat (double 1.000000e+00)
}

__SVFloat64_t splat_float_imm___SVFloat64_t(void) {
  return __builtin_splatvector(1.0f, __SVFloat64_t);
  // CHECK-LABEL: @splat_float_imm___SVFloat64_t
  // CHECK: ret <vscale x {{[0-9]*}} x double> splat (double 1.000000e+00)
}

__SVFloat64_t splat_double_imm___SVFloat64_t(void) {
  return __builtin_splatvector(1.0, __SVFloat64_t);
  // CHECK-LABEL: @splat_double_imm___SVFloat64_t
  // CHECK: ret <vscale x {{[0-9]*}} x double> splat (double 1.000000e+00)
}

__SVFloat64_t splat_bool_var___SVFloat64_t(bool x) {
  return __builtin_splatvector(x, __SVFloat64_t);
  // CHECK-LABEL: @splat_bool_var___SVFloat64_t
  // CHECK: %[[STOREDV:.*]] = zext i1 %x to i8
  // CHECK: store i8 %[[STOREDV]], ptr %[[WHERE:.*]]
  // CHECK: %[[WHAT:.*]] = load i8, ptr %[[WHERE]]
  // CHECK: %[[WHAT_I1:.*]] = icmp ne i8 %[[WHAT]], 0
  // CHECK: %[[INSERT:.*]] = insertelement <vscale x {{[0-9]*}} x i1> poison, i1 %[[WHAT_I1]], i64 0
  // CHECK: %[[SPLAT:.*]] = shufflevector <vscale x {{[0-9]*}} x i1> %[[INSERT]], <vscale x {{[0-9]*}} x i1> poison, <vscale x {{[0-9]*}} x i32> zeroinitializer
  // CHECK: %[[CONV:.*]] = uitofp <vscale x {{[0-9]*}} x i1> %[[SPLAT]] to <vscale x {{[0-9]*}} x double>
  // CHECK: ret <vscale x {{[0-9]*}} x double> %[[CONV]]
}

__SVFloat64_t splat_int_var___SVFloat64_t(int x) {
  return __builtin_splatvector(x, __SVFloat64_t);
  // CHECK-LABEL: @splat_int_var___SVFloat64_t
  // CHECK: store i32 %x, ptr %[[WHERE:.*]]
  // CHECK: %[[WHAT:.*]] = load i32, ptr %[[WHERE]]
  // CHECK: %[[INSERT:.*]] = insertelement <vscale x {{[0-9]*}} x i32> poison, i32 %[[WHAT]], i64 0
  // CHECK: %[[SPLAT:.*]] = shufflevector <vscale x {{[0-9]*}} x i32> %[[INSERT]], <vscale x {{[0-9]*}} x i32> poison, <vscale x {{[0-9]*}} x i32> zeroinitializer
  // CHECK: %[[CONV:.*]] = sitofp <vscale x {{[0-9]*}} x i32> %[[SPLAT]] to <vscale x {{[0-9]*}} x double>
  // CHECK: ret <vscale x {{[0-9]*}} x double> %[[CONV]]
}

__SVFloat64_t splat_uint_var___SVFloat64_t(unsigned x) {
  return __builtin_splatvector(x, __SVFloat64_t);
  // CHECK-LABEL: @splat_uint_var___SVFloat64_t
  // CHECK: store i32 %x, ptr %[[WHERE:.*]]
  // CHECK: %[[WHAT:.*]] = load i32, ptr %[[WHERE]]
  // CHECK: %[[INSERT:.*]] = insertelement <vscale x {{[0-9]*}} x i32> poison, i32 %[[WHAT]], i64 0
  // CHECK: %[[SPLAT:.*]] = shufflevector <vscale x {{[0-9]*}} x i32> %[[INSERT]], <vscale x {{[0-9]*}} x i32> poison, <vscale x {{[0-9]*}} x i32> zeroinitializer
  // CHECK: %[[CONV:.*]] = uitofp <vscale x {{[0-9]*}} x i32> %[[SPLAT]] to <vscale x {{[0-9]*}} x double>
  // CHECK: ret <vscale x {{[0-9]*}} x double> %[[CONV]]
}

__SVFloat64_t splat_float_var___SVFloat64_t(float x) {
  return __builtin_splatvector(x, __SVFloat64_t);
  // CHECK-LABEL: @splat_float_var___SVFloat64_t
  // CHECK: store float %x, ptr %[[WHERE:.*]]
  // CHECK: %[[WHAT:.*]] = load float, ptr %[[WHERE]]
  // CHECK: %[[INSERT:.*]] = insertelement <vscale x {{[0-9]*}} x float> poison, float %[[WHAT]], i64 0
  // CHECK: %[[SPLAT:.*]] = shufflevector <vscale x {{[0-9]*}} x float> %[[INSERT]], <vscale x {{[0-9]*}} x float> poison, <vscale x {{[0-9]*}} x i32> zeroinitializer
  // CHECK: %[[CONV:.*]] = fpext <vscale x {{[0-9]*}} x float> %[[SPLAT]] to <vscale x {{[0-9]*}} x double>
  // CHECK: ret <vscale x {{[0-9]*}} x double> %[[CONV]]
}

__SVFloat64_t splat_double_var___SVFloat64_t(double x) {
  return __builtin_splatvector(x, __SVFloat64_t);
  // CHECK-LABEL: @splat_double_var___SVFloat64_t
  // CHECK: store double %x, ptr %[[WHERE:.*]]
  // CHECK: %[[WHAT:.*]] = load double, ptr %[[WHERE]]
  // CHECK: %[[INSERT:.*]] = insertelement <vscale x {{[0-9]*}} x double> poison, double %[[WHAT]], i64 0
  // CHECK: %[[SPLAT:.*]] = shufflevector <vscale x {{[0-9]*}} x double> %[[INSERT]], <vscale x {{[0-9]*}} x double> poison, <vscale x {{[0-9]*}} x i32> zeroinitializer
  // CHECK: ret <vscale x {{[0-9]*}} x double> %[[SPLAT]]
}

__SVFloat32_t splat_flt_trunc(double x) {
  return __builtin_splatvector(x, __SVFloat32_t);
  // CHECK-LABEL: @splat_flt_trunc
  // CHECK: fptrunc <vscale x {{[0-9]*}} x double> %{{.*}} to <vscale x {{[0-9]*}} x float>
}

__SVFloat64_t splat_flt_ext(float x) {
  return __builtin_splatvector(x, __SVFloat64_t);
  // CHECK-LABEL: @splat_flt_ext
  // CHECK: fpext <vscale x {{[0-9]*}} x float> %{{.*}} to <vscale x {{[0-9]*}} x double>
}

__SVInt64_t splat_flt_tosi(float x) {
  return __builtin_splatvector(x, __SVInt64_t);
  // CHECK-LABEL: @splat_flt_tosi
  // CHECK: fptosi <vscale x {{[0-9]*}} x float> %{{.*}} to <vscale x {{[0-9]*}} x i64>
}

__SVUint64_t splat_flt_toui(float x) {
  return __builtin_splatvector(x, __SVUint64_t);
  // CHECK-LABEL: @splat_flt_toui
  // CHECK: fptoui <vscale x {{[0-9]*}} x float> %{{.*}} to <vscale x {{[0-9]*}} x i64>
}

__SVUint64_t splat_fltd_toui(double x) {
  return __builtin_splatvector(x, __SVUint64_t);
  // CHECK-LABEL: @splat_fltd_toui
  // CHECK: fptoui <vscale x {{[0-9]*}} x double> %{{.*}} to <vscale x {{[0-9]*}} x i64>
}

__SVUint64_t splat_int_zext(unsigned short x) {
  return __builtin_splatvector(x, __SVUint64_t);
  // CHECK-LABEL: @splat_int_zext
  // CHECK: zext <vscale x {{[0-9]*}} x i16> %{{.*}} to <vscale x {{[0-9]*}} x i64>
}

__SVInt64_t splat_int_sext(short x) {
  return __builtin_splatvector(x, __SVInt64_t);
  // CHECK-LABEL: @splat_int_sext
  // CHECK: sext <vscale x {{[0-9]*}} x i16> %{{.*}} to <vscale x {{[0-9]*}} x i64>
}

__SVFloat32_t splat_int_tofp(short x) {
  return __builtin_splatvector(x, __SVFloat32_t);
  // CHECK-LABEL: @splat_int_tofp
  // CHECK: sitofp <vscale x {{[0-9]*}} x i16> %{{.*}} to <vscale x {{[0-9]*}} x float>
}

__SVFloat32_t splat_uint_tofp(unsigned short x) {
  return __builtin_splatvector(x, __SVFloat32_t);
  // CHECK-LABEL: @splat_uint_tofp
  // CHECK: uitofp <vscale x {{[0-9]*}} x i16> %{{.*}} to <vscale x {{[0-9]*}} x float>
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
  __SVFloat64_t splat_int_toT_fp(long x) {
    // CHECK-LABEL: @splat_int_toT_fp
    // CHECK: sitofp <vscale x {{[0-9]*}} x i64> %{{.*}} to <vscale x {{[0-9]*}} x double>
    return splat_int_toT<__SVFloat64_t>(x);
  }
}
#else
__SVFloat64_t splat_int_toT_fp(long x) {
  return __builtin_splatvector(x, __SVFloat64_t);
}
#endif
