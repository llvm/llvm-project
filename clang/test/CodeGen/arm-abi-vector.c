// RUN: %clang_cc1 -triple armv7-apple-darwin -target-abi aapcs -emit-llvm -o - %s | FileCheck %s
// RUN: %clang_cc1 -triple armv7-apple-darwin -target-abi apcs-gnu -emit-llvm -o - %s | FileCheck -check-prefix=APCS-GNU %s
// RUN: %clang_cc1 -triple arm-linux-androideabi -emit-llvm -o - %s | FileCheck -check-prefix=ANDROID %s

#include <stdarg.h>

typedef __attribute__(( ext_vector_type(2) ))  int __int2;
typedef __attribute__(( ext_vector_type(3) ))  char __char3;
typedef __attribute__(( ext_vector_type(5) ))  char __char5;
typedef __attribute__(( ext_vector_type(9) ))  char __char9;
typedef __attribute__(( ext_vector_type(19) )) char __char19;
typedef __attribute__(( ext_vector_type(3) ))  short __short3;
typedef __attribute__(( ext_vector_type(5) ))  short __short5;

// Passing legal vector types as varargs.
double varargs_vec_2i(int fixed, ...) {
// CHECK: varargs_vec_2i
// CHECK: [[VAR:%.*]] = alloca <2 x i32>, align 8
// CHECK: [[AP_ALIGN:%.*]] = call ptr @llvm.ptrmask.p0.i32(ptr {{%.*}}, i32 -8)
// CHECK: [[AP_NEXT:%.*]] = getelementptr inbounds i8, ptr [[AP_ALIGN]], i32 8
// CHECK: [[VEC:%.*]] = load <2 x i32>, ptr [[AP_ALIGN]], align 8
// CHECK: store <2 x i32> [[VEC]], ptr [[VAR]], align 8
// APCS-GNU: varargs_vec_2i
// APCS-GNU: [[VAR:%.*]] = alloca <2 x i32>, align 8
// APCS-GNU: [[AP:%.*]] = load ptr,
// APCS-GNU: [[AP_NEXT:%.*]] = getelementptr inbounds i8, ptr [[AP]], i32 8
// APCS-GNU: [[VEC:%.*]] = load <2 x i32>, ptr [[AP]], align 4
// APCS-GNU: store <2 x i32> [[VEC]], ptr [[VAR]], align 8
// ANDROID: varargs_vec_2i
// ANDROID: [[VAR:%.*]] = alloca <2 x i32>, align 8
// ANDROID: [[AP_ALIGN:%.*]] = call ptr @llvm.ptrmask.p0.i32(ptr {{%.*}}, i32 -8)
// ANDROID: [[AP_NEXT:%.*]] = getelementptr inbounds i8, ptr [[AP_ALIGN]], i32 8
// ANDROID: [[VEC:%.*]] = load <2 x i32>, ptr [[AP_ALIGN]], align 8
// ANDROID: store <2 x i32> [[VEC]], ptr [[VAR]], align 8
  va_list ap;
  double sum = fixed;
  va_start(ap, fixed);
  __int2 c3 = va_arg(ap, __int2);
  sum = sum + c3.x + c3.y;
  va_end(ap);
  return sum;
}

double test_2i(__int2 *in) {
// CHECK: test_2i
// CHECK: call arm_aapcscc double (i32, ...) @varargs_vec_2i(i32 noundef 3, <2 x i32> noundef {{%.*}})
// APCS-GNU: test_2i
// APCS-GNU: call double (i32, ...) @varargs_vec_2i(i32 noundef 3, <2 x i32> noundef {{%.*}})
// ANDROID: test_2i
// ANDROID: call double (i32, ...) @varargs_vec_2i(i32 noundef 3, <2 x i32> noundef {{%.*}})
  return varargs_vec_2i(3, *in);
}

double varargs_vec_3c(int fixed, ...) {
// CHECK: varargs_vec_3c
// CHECK: alloca <3 x i8>, align 4
// CHECK: [[AP_NEXT:%.*]] = getelementptr inbounds i8, ptr [[AP:%.*]], i32 4
// APCS-GNU: varargs_vec_3c
// APCS-GNU: alloca <3 x i8>, align 4
// APCS-GNU: [[AP_NEXT:%.*]] = getelementptr inbounds i8, ptr [[AP:%.*]], i32 4
// ANDROID: varargs_vec_3c
// ANDROID: alloca <3 x i8>, align 4
// ANDROID: [[AP_NEXT:%.*]] = getelementptr inbounds i8, ptr [[AP:%.*]], i32 4
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
// CHECK: call arm_aapcscc double (i32, ...) @varargs_vec_3c(i32 noundef 3, i32 {{%.*}})
// APCS-GNU: test_3c
// APCS-GNU: call double (i32, ...) @varargs_vec_3c(i32 noundef 3, i32 {{%.*}})
// ANDROID: test_3c
// ANDROID: call double (i32, ...) @varargs_vec_3c(i32 noundef 3, <3 x i8> noundef {{%.*}})
  return varargs_vec_3c(3, *in);
}

double varargs_vec_5c(int fixed, ...) {
// CHECK: varargs_vec_5c
// CHECK: [[VAR:%.*]] = alloca <5 x i8>, align 8
// CHECK: [[AP_ALIGN:%.*]] = call ptr @llvm.ptrmask.p0.i32(ptr {{%.*}}, i32 -8)
// CHECK: [[AP_NEXT:%.*]] = getelementptr inbounds i8, ptr [[AP_ALIGN]], i32 8
// CHECK: [[VEC:%.*]] = load <5 x i8>, ptr [[AP_ALIGN]], align 8
// CHECK: store <5 x i8> [[VEC]], ptr [[VAR]], align 8
// APCS-GNU: varargs_vec_5c
// APCS-GNU: [[VAR:%.*]] = alloca <5 x i8>, align 8
// APCS-GNU: [[AP:%.*]] = load ptr,
// APCS-GNU: [[AP_NEXT:%.*]] = getelementptr inbounds i8, ptr [[AP]], i32 8
// APCS-GNU: [[VEC:%.*]] = load <5 x i8>, ptr [[AP]], align 4
// APCS-GNU: store <5 x i8> [[VEC]], ptr [[VAR]], align 8
// ANDROID: varargs_vec_5c
// ANDROID: [[VAR:%.*]] = alloca <5 x i8>, align 8
// ANDROID: [[AP_ALIGN:%.*]] = call ptr @llvm.ptrmask.p0.i32(ptr {{%.*}}, i32 -8)
// ANDROID: [[AP_NEXT:%.*]] = getelementptr inbounds i8, ptr [[AP_ALIGN]], i32 8
// ANDROID: [[VEC:%.*]] = load <5 x i8>, ptr [[AP_ALIGN]], align 8
// ANDROID: store <5 x i8> [[VEC]], ptr [[VAR]], align 8
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
// CHECK: call arm_aapcscc double (i32, ...) @varargs_vec_5c(i32 noundef 5, <2 x i32> {{%.*}})
// APCS-GNU: test_5c
// APCS-GNU: call double (i32, ...) @varargs_vec_5c(i32 noundef 5, <2 x i32> {{%.*}})
// ANDROID: test_5c
// ANDROID: call double (i32, ...) @varargs_vec_5c(i32 noundef 5, <2 x i32> {{%.*}})
  return varargs_vec_5c(5, *in);
}

double varargs_vec_9c(int fixed, ...) {
// CHECK: varargs_vec_9c
// CHECK: [[VAR:%.*]] = alloca <9 x i8>, align 16
// CHECK: [[AP_ALIGN:%.*]] = call ptr @llvm.ptrmask.p0.i32(ptr {{%.*}}, i32 -8)
// CHECK: [[AP_NEXT:%.*]] = getelementptr inbounds i8, ptr [[AP_ALIGN]], i32 16
// CHECK: [[T0:%.*]] = load <9 x i8>, ptr [[AP_ALIGN]], align 8
// CHECK: store <9 x i8> [[T0]], ptr [[VAR]], align 16
// APCS-GNU: varargs_vec_9c
// APCS-GNU: [[VAR:%.*]] = alloca <9 x i8>, align 16
// APCS-GNU: [[AP:%.*]] = load ptr,
// APCS-GNU: [[AP_NEXT:%.*]] = getelementptr inbounds i8, ptr [[AP]], i32 16
// APCS-GNU: [[VEC:%.*]] = load <9 x i8>, ptr [[AP]], align 4
// APCS-GNU: store <9 x i8> [[VEC]], ptr [[VAR]], align 16
// ANDROID: varargs_vec_9c
// ANDROID: [[VAR:%.*]] = alloca <9 x i8>, align 16
// ANDROID: [[AP_ALIGN:%.*]] = call ptr @llvm.ptrmask.p0.i32(ptr {{%.*}}, i32 -8)
// ANDROID: [[AP_NEXT:%.*]] = getelementptr inbounds i8, ptr [[AP_ALIGN]], i32 16
// ANDROID: [[T0:%.*]] = load <9 x i8>, ptr [[AP_ALIGN]], align 8
// ANDROID: store <9 x i8> [[T0]], ptr [[VAR]], align 16
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
// CHECK: call arm_aapcscc double (i32, ...) @varargs_vec_9c(i32 noundef 9, <4 x i32> {{%.*}})
// APCS-GNU: test_9c
// APCS-GNU: call double (i32, ...) @varargs_vec_9c(i32 noundef 9, <4 x i32> {{%.*}})
// ANDROID: test_9c
// ANDROID: call double (i32, ...) @varargs_vec_9c(i32 noundef 9, <4 x i32> {{%.*}})
  return varargs_vec_9c(9, *in);
}

double varargs_vec_19c(int fixed, ...) {
// CHECK: varargs_vec_19c
// CHECK: [[AP_NEXT:%.*]] = getelementptr inbounds i8, ptr [[AP:%.*]], i32 4
// CHECK: [[VAR2:%.*]] = load ptr, ptr [[AP]]
// APCS-GNU: varargs_vec_19c
// APCS-GNU: [[AP_NEXT:%.*]] = getelementptr inbounds i8, ptr [[AP:%.*]], i32 4
// APCS-GNU: [[VAR2:%.*]] = load ptr, ptr [[AP]]
// ANDROID: varargs_vec_19c
// ANDROID: [[AP_NEXT:%.*]] = getelementptr inbounds i8, ptr [[AP:%.*]], i32 4
// ANDROID: [[VAR2:%.*]] = load ptr, ptr [[AP]]
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
// CHECK: call arm_aapcscc double (i32, ...) @varargs_vec_19c(i32 noundef 19, ptr noundef {{%.*}})
// APCS-GNU: test_19c
// APCS-GNU: call double (i32, ...) @varargs_vec_19c(i32 noundef 19, ptr noundef {{%.*}})
// ANDROID: test_19c
// ANDROID: call double (i32, ...) @varargs_vec_19c(i32 noundef 19, ptr noundef {{%.*}})
  return varargs_vec_19c(19, *in);
}

double varargs_vec_3s(int fixed, ...) {
// CHECK: varargs_vec_3s
// CHECK: alloca <3 x i16>, align 8
// CHECK: [[AP_ALIGN:%.*]] = call ptr @llvm.ptrmask.p0.i32(ptr {{%.*}}, i32 -8)
// CHECK: [[AP_NEXT:%.*]] = getelementptr inbounds i8, ptr [[AP_ALIGN]], i32 8
// APCS-GNU: varargs_vec_3s
// APCS-GNU: [[VAR:%.*]] = alloca <3 x i16>, align 8
// APCS-GNU: [[AP:%.*]] = load ptr,
// APCS-GNU: [[AP_NEXT:%.*]] = getelementptr inbounds i8, ptr [[AP]], i32 8
// APCS-GNU: [[VEC:%.*]] = load <3 x i16>, ptr [[AP]], align 4
// ANDROID: varargs_vec_3s
// ANDROID: alloca <3 x i16>, align 8
// ANDROID: [[AP_ALIGN:%.*]] = call ptr @llvm.ptrmask.p0.i32(ptr {{%.*}}, i32 -8)
// ANDROID: [[AP_NEXT:%.*]] = getelementptr inbounds i8, ptr [[AP_ALIGN]], i32 8
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
// CHECK: call arm_aapcscc double (i32, ...) @varargs_vec_3s(i32 noundef 3, <2 x i32> {{%.*}})
// APCS-GNU: test_3s
// APCS-GNU: call double (i32, ...) @varargs_vec_3s(i32 noundef 3, <2 x i32> {{%.*}})
// ANDROID: test_3s
// ANDROID: call double (i32, ...) @varargs_vec_3s(i32 noundef 3, <3 x i16> noundef {{%.*}})
  return varargs_vec_3s(3, *in);
}

double varargs_vec_5s(int fixed, ...) {
// CHECK: varargs_vec_5s
// CHECK: [[VAR_ALIGN:%.*]] = alloca <5 x i16>, align 16
// CHECK: [[AP_ALIGN:%.*]] = call ptr @llvm.ptrmask.p0.i32(ptr {{%.*}}, i32 -8)
// CHECK: [[AP_NEXT:%.*]] = getelementptr inbounds i8, ptr [[AP_ALIGN]], i32 16
// CHECK: [[VEC:%.*]] = load <5 x i16>, ptr [[AP_ALIGN]], align 8
// CHECK: store <5 x i16> [[VEC]], ptr [[VAR_ALIGN]], align 16
// APCS-GNU: varargs_vec_5s
// APCS-GNU: [[VAR:%.*]] = alloca <5 x i16>, align 16
// APCS-GNU: [[AP:%.*]] = load ptr,
// APCS-GNU: [[AP_NEXT:%.*]] = getelementptr inbounds i8, ptr [[AP]], i32 16
// APCS-GNU: [[VEC:%.*]] = load <5 x i16>, ptr [[AP]], align 4
// ANDROID: varargs_vec_5s
// ANDROID: [[VAR_ALIGN:%.*]] = alloca <5 x i16>, align 16
// ANDROID: [[AP_ALIGN:%.*]] = call ptr @llvm.ptrmask.p0.i32(ptr {{%.*}}, i32 -8)
// ANDROID: [[AP_NEXT:%.*]] = getelementptr inbounds i8, ptr [[AP_ALIGN]], i32 16
// ANDROID: [[VEC:%.*]] = load <5 x i16>, ptr [[AP_ALIGN]], align 8
// ANDROID: store <5 x i16> [[VEC]], ptr [[VAR_ALIGN]], align 16
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
// CHECK: call arm_aapcscc double (i32, ...) @varargs_vec_5s(i32 noundef 5, <4 x i32> {{%.*}})
// APCS-GNU: test_5s
// APCS-GNU: call double (i32, ...) @varargs_vec_5s(i32 noundef 5, <4 x i32> {{%.*}})
// ANDROID: test_5s
// ANDROID: call double (i32, ...) @varargs_vec_5s(i32 noundef 5, <4 x i32> {{%.*}})
  return varargs_vec_5s(5, *in);
}

// Pass struct as varargs.
typedef struct
{
  __int2 i2;
  float f;
} StructWithVec;

double varargs_struct(int fixed, ...) {
// CHECK: varargs_struct
// CHECK: [[AP_ALIGN:%.*]] = call ptr @llvm.ptrmask.p0.i32(ptr {{%.*}}, i32 -8)
// CHECK: [[AP_NEXT:%.*]] = getelementptr inbounds i8, ptr [[AP_ALIGN]], i32 16
// APCS-GNU: varargs_struct
// APCS-GNU: [[VAR_ALIGN:%.*]] = alloca %struct.StructWithVec
// APCS-GNU: [[AP_NEXT:%.*]] = getelementptr inbounds i8, ptr {{%.*}}, i32 16
// APCS-GNU: call void @llvm.memcpy
// ANDROID: varargs_struct
// ANDROID: [[AP_ALIGN:%.*]] = call ptr @llvm.ptrmask.p0.i32(ptr {{%.*}}, i32 -8)
// ANDROID: [[AP_NEXT:%.*]] = getelementptr inbounds i8, ptr [[AP_ALIGN]], i32 16
  va_list ap;
  double sum = fixed;
  va_start(ap, fixed);
  StructWithVec c3 = va_arg(ap, StructWithVec);
  sum = sum + c3.i2.x + c3.i2.y + c3.f;
  va_end(ap);
  return sum;
}

double test_struct(StructWithVec* d) {
// CHECK: test_struct
// CHECK: call arm_aapcscc double (i32, ...) @varargs_struct(i32 noundef 3, [2 x i64] {{%.*}})
// APCS-GNU: test_struct
// APCS-GNU: call double (i32, ...) @varargs_struct(i32 noundef 3, [2 x i64] {{%.*}})
// ANDROID: test_struct
// ANDROID: call double (i32, ...) @varargs_struct(i32 noundef 3, [2 x i64] {{%.*}})
  return varargs_struct(3, *d);
}
