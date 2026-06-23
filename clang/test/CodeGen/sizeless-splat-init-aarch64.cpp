// RUN: %clang_cc1 -triple aarch64 -target-feature +sve %s -emit-llvm -o - | FileCheck %s

#include <stdint.h>

// CHECK-LABEL: __SVBool_t_fun_1
// CHECK: store <vscale x {{[0-9]*}} x i1> splat (i1 true), ptr %[[RETPTR:.*]]
// CHECK: %[[RETVAL:.*]] = load <vscale x {{[0-9]*}} x i1>, ptr %[[RETPTR]]
// CHECK: ret <vscale x {{[0-9]*}} x i1> %[[RETVAL]]
__SVBool_t __SVBool_t_fun_1(void) {
  __SVBool_t svi1(true);
  return svi1;
}

// CHECK-LABEL: __SVBool_t_fun_2
// CHECK: store <vscale x {{[0-9]*}} x i1> splat (i1 true), ptr %[[RETPTR:.*]]
// CHECK: %[[RETVAL:.*]] = load <vscale x {{[0-9]*}} x i1>, ptr %[[RETPTR]]
// CHECK: ret <vscale x {{[0-9]*}} x i1> %[[RETVAL]]
__SVBool_t __SVBool_t_fun_2(void) {
  __SVBool_t svi1 = true;
  return svi1;
}

// CHECK-LABEL: __SVBool_t_argfun_1
// CHECK: %[[STOREDV:.*]] = zext i1 %arg to i8
// CHECK: store i8 %[[STOREDV]], ptr %[[WHERE:.*]]
// CHECK: %[[WHAT:.*]] = load i8, ptr %[[WHERE]]
// CHECK: %[[WHAT_I1:.*]] = icmp ne i8 %[[WHAT]], 0
// CHECK: %[[INSERT:.*]] = insertelement <vscale x {{[0-9]*}} x i1> poison, i1 %[[WHAT_I1]], i64 0
// CHECK: %[[SPLAT:.*]] = shufflevector <vscale x {{[0-9]*}} x i1> %[[INSERT]], <vscale x {{[0-9]*}} x i1> poison, <vscale x {{[0-9]*}} x i32> zeroinitializer
// CHECK: store <vscale x {{[0-9]*}} x i1> %[[SPLAT]], ptr %[[RETPTR:.*]]
// CHECK: %[[RETVAL:.*]] = load <vscale x {{[0-9]*}} x i1>, ptr %[[RETPTR]]
// CHECK: ret <vscale x {{[0-9]*}} x i1> %[[RETVAL]]
__SVBool_t __SVBool_t_argfun_1(bool arg) {
  __SVBool_t svi1(arg);
  return svi1;
}

void __SVBool_t_sink(__SVBool_t);

// CHECK-LABEL: __SVBool_t_argfun_2
// CHECK: store i32 %i, ptr %[[WHERE:.*]]
// CHECK: %[[WHAT:.*]] = load i32, ptr %[[WHERE]]
// CHECK: %[[WHAT_I1:.*]] = icmp ne i32 %[[WHAT]], 0
// CHECK: %[[INSERT:.*]] = insertelement <vscale x {{[0-9]*}} x i1> poison, i1 %[[WHAT_I1]], i64 0
// CHECK: %[[SPLAT:.*]] = shufflevector <vscale x {{[0-9]*}} x i1> %[[INSERT]], <vscale x {{[0-9]*}} x i1> poison, <vscale x {{[0-9]*}} x i32> zeroinitializer
// CHECK: call void @[[SINK:.*]](<vscale x {{[0-9]*}} x i1> %[[SPLAT]])
// CHECK: ret void
void __SVBool_t_argfun_2(int i)
{
  __SVBool_t_sink(i);
}

// CHECK-LABEL: __SVInt8_t_fun_1
// CHECK: store <vscale x {{[0-9]*}} x i8> splat (i8 123), ptr %[[RETPTR:.*]]
// CHECK: %[[RETVAL:.*]] = load <vscale x {{[0-9]*}} x i8>, ptr %[[RETPTR]]
// CHECK: ret <vscale x {{[0-9]*}} x i8> %[[RETVAL]]
__SVInt8_t __SVInt8_t_fun_1(void) {
  __SVInt8_t svi8(123);
  return svi8;
}

// CHECK-LABEL: __SVInt8_t_fun_2
// CHECK: store <vscale x {{[0-9]*}} x i8> splat (i8 123), ptr %[[RETPTR:.*]]
// CHECK: %[[RETVAL:.*]] = load <vscale x {{[0-9]*}} x i8>, ptr %[[RETPTR]]
// CHECK: ret <vscale x {{[0-9]*}} x i8> %[[RETVAL]]
__SVInt8_t __SVInt8_t_fun_2(void) {
  __SVInt8_t svi8 = 123;
  return svi8;
}

// CHECK-LABEL: __SVInt8_t_argfun_1
// CHECK: store i8 %arg, ptr %[[WHERE:.*]]
// CHECK: %[[WHAT:.*]] = load i8, ptr %[[WHERE]]
// CHECK: %[[INSERT:.*]] = insertelement <vscale x {{[0-9]*}} x i8> poison, i8 %[[WHAT]], i64 0
// CHECK: %[[SPLAT:.*]] = shufflevector <vscale x {{[0-9]*}} x i8> %[[INSERT]], <vscale x {{[0-9]*}} x i8> poison, <vscale x {{[0-9]*}} x i32> zeroinitializer
// CHECK: store <vscale x {{[0-9]*}} x i8> %[[SPLAT]], ptr %[[RETPTR:.*]]
// CHECK: %[[RETVAL:.*]] = load <vscale x {{[0-9]*}} x i8>, ptr %[[RETPTR]]
// CHECK: ret <vscale x {{[0-9]*}} x i8> %[[RETVAL]]
__SVInt8_t __SVInt8_t_argfun_1(int8_t arg) {
  __SVInt8_t svi8(arg);
  return svi8;
}

void __SVInt8_t_sink(__SVInt8_t);

// CHECK-LABEL __SVInt8_t_argfun_2
// CHECK: store i32 %i, ptr %[[WHERE:.*]]
// CHECK: %[[WHAT:.*]] = load i32, ptr %[[WHERE]]
// CHECK: %[[WHAT_I8:.*]] = trunc i32 %[[WHAT]] to i8
// CHECK: %[[INSERT:.*]] = insertelement <vscale x {{[0-9]*}} x i8> poison, i8 %[[WHAT_I8]], i64 0
// CHECK: %[[SPLAT:.*]] = shufflevector <vscale x {{[0-9]*}} x i8> %[[INSERT]], <vscale x {{[0-9]*}} x i8> poison, <vscale x {{[0-9]*}} x i32> zeroinitializer
// CHECK: call void @[[SINK:.*]](<vscale x {{[0-9]*}} x i8> %[[SPLAT]])
// CHECK: ret void
void __SVInt8_t_argfun_2(int i)
{
  __SVInt8_t_sink(i);
}

// CHECK-LABEL: __SVUint32_t_fun_1
// CHECK: store <vscale x {{[0-9]*}} x i32> splat (i32 1024), ptr %[[RETPTR:.*]]
// CHECK: %[[RETVAL:.*]] = load <vscale x {{[0-9]*}} x i32>, ptr %[[RETPTR]]
// CHECK: ret <vscale x {{[0-9]*}} x i32> %[[RETVAL]]
__SVUint32_t __SVUint32_t_fun_1(void) {
  __SVUint32_t svui32(1024U);
  return svui32;
}

// CHECK-LABEL: __SVUint32_t_fun_2
// CHECK: store <vscale x {{[0-9]*}} x i32> splat (i32 1024), ptr %[[RETPTR:.*]]
// CHECK: %[[RETVAL:.*]] = load <vscale x {{[0-9]*}} x i32>, ptr %[[RETPTR]]
// CHECK: ret <vscale x {{[0-9]*}} x i32> %[[RETVAL]]
__SVUint32_t __SVUint32_t_fun_2(void) {
  __SVUint32_t svui32 = 1024U;
  return svui32;
}

// CHECK-LABEL: __SVUint32_t_argfun_1
// CHECK: store i32 %arg, ptr %[[WHERE:.*]]
// CHECK: %[[WHAT:.*]] = load i32, ptr %[[WHERE]]
// CHECK: %[[INSERT:.*]] = insertelement <vscale x {{[0-9]*}} x i32> poison, i32 %[[WHAT]], i64 0
// CHECK: %[[SPLAT:.*]] = shufflevector <vscale x {{[0-9]*}} x i32> %[[INSERT]], <vscale x {{[0-9]*}} x i32> poison, <vscale x {{[0-9]*}} x i32> zeroinitializer
// CHECK: store <vscale x {{[0-9]*}} x i32> %[[SPLAT]], ptr %[[RETPTR:.*]]
// CHECK: %[[RETVAL:.*]] = load <vscale x {{[0-9]*}} x i32>, ptr %[[RETPTR]]
// CHECK: ret <vscale x {{[0-9]*}} x i32> %[[RETVAL]]
__SVUint32_t __SVUint32_t_argfun_1(uint32_t arg) {
  __SVUint32_t svui32(arg);
  return svui32;
}

void __SVUint32_t_sink(__SVUint32_t);

// CHECK-LABEL: __SVUint32_t_argfun_2
// CHECK: store i32 %i, ptr %[[WHERE:.*]]
// CHECK: %[[WHAT:.*]] = load i32, ptr %[[WHERE]]
// CHECK: %[[INSERT:.*]] = insertelement <vscale x {{[0-9]*}} x i32> poison, i32 %[[WHAT]], i64 0
// CHECK: %[[SPLAT:.*]] = shufflevector <vscale x {{[0-9]*}} x i32> %[[INSERT]], <vscale x {{[0-9]*}} x i32> poison, <vscale x {{[0-9]*}} x i32> zeroinitializer
// CHECK: call void @[[SINK:.*]](<vscale x {{[0-9]*}} x i32> %[[SPLAT]])
// CHECK: ret void
void __SVUint32_t_argfun_2(int i)
{
  __SVUint32_t_sink(i);
}

// CHECK-LABEL: __SVFloat32_t_fun_1
// CHECK: store <vscale x {{[0-9]*}} x float> splat (float 1.230000e-01), ptr %[[RETPTR:.*]]
// CHECK: %[[RETVAL:.*]] = load <vscale x {{[0-9]*}} x float>, ptr %[[RETPTR]]
// CHECK: ret <vscale x {{[0-9]*}} x float> %[[RETVAL]]
__SVFloat32_t __SVFloat32_t_fun_1(void) {
  __SVFloat32_t svfloat32(0.123);
  return svfloat32;
}

// CHECK-LABEL: __SVFloat32_t_fun_2
// CHECK: store <vscale x {{[0-9]*}} x float> splat (float 1.230000e-01), ptr %[[RETPTR:.*]]
// CHECK: %[[RETVAL:.*]] = load <vscale x {{[0-9]*}} x float>, ptr %[[RETPTR]]
// CHECK: ret <vscale x {{[0-9]*}} x float> %[[RETVAL]]
__SVFloat32_t __SVFloat32_t_fun_2(void) {
  __SVFloat32_t svfloat32 = 0.123;
  return svfloat32;
}

// CHECK-LABEL: __SVFloat32_t_argfun_1
// CHECK: store float %arg, ptr %[[WHERE:.*]]
// CHECK: %[[WHAT:.*]] = load float, ptr %[[WHERE]]
// CHECK: %[[INSERT:.*]] = insertelement <vscale x {{[0-9]*}} x float> poison, float %[[WHAT]], i64 0
// CHECK: %[[SPLAT:.*]] = shufflevector <vscale x {{[0-9]*}} x float> %[[INSERT]], <vscale x {{[0-9]*}} x float> poison, <vscale x {{[0-9]*}} x i32> zeroinitializer
// CHECK: store <vscale x {{[0-9]*}} x float> %[[SPLAT]], ptr %[[RETPTR:.*]]
// CHECK: %[[RETVAL:.*]] = load <vscale x {{[0-9]*}} x float>, ptr %[[RETPTR]]
// CHECK: ret <vscale x {{[0-9]*}} x float> %[[RETVAL]]
__SVFloat32_t __SVFloat32_t_argfun_1(float arg) {
  __SVFloat32_t svfloat32(arg);
  return svfloat32;
}

void __SVFloat32_t_sink(__SVFloat32_t);

// CHECK-LABEL: __SVFloat32_t_argfun_2
// CHECK: store i32 %i, ptr %[[WHERE:.*]]
// CHECK: %[[WHAT:.*]] = load i32, ptr %[[WHERE]]
// CHECK: %[[WHAT_FP:.*]] = sitofp i32 %[[WHAT]] to float
// CHECK: %[[INSERT:.*]] = insertelement <vscale x {{[0-9]*}} x float> poison, float %[[WHAT_FP]], i64 0
// CHECK: %[[SPLAT:.*]] = shufflevector <vscale x {{[0-9]*}} x float> %[[INSERT]], <vscale x {{[0-9]*}} x float> poison, <vscale x {{[0-9]*}} x i32> zeroinitializer
// CHECK: call void @[[SINK:.*]](<vscale x {{[0-9]*}} x float> %[[SPLAT]])
// CHECK: ret void
void __SVFloat32_t_argfun_2(int i)
{
  __SVFloat32_t_sink(i);
}
