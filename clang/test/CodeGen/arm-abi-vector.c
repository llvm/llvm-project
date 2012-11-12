// RUN: %clang_cc1 -triple armv7-apple-darwin -target-abi aapcs -emit-llvm -o - %s | FileCheck %s
// RUN: %clang_cc1 -triple armv7-apple-darwin -target-abi apcs-gnu -emit-llvm -o - %s | FileCheck -check-prefix=APCS-GNU %s

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
// CHECK: alloca <2 x i32>, align 8
// CHECK: [[ALIGN:%.*]] = and i32 [[VAR:%.*]], -8
// CHECK: [[AP_ALIGN:%.*]] = inttoptr i32 [[ALIGN]] to i8*
// CHECK: [[AP_NEXT:%.*]] = getelementptr i8* [[AP_ALIGN]], i32 8
// CHECK: bitcast i8* [[AP_ALIGN]] to <2 x i32>*
// APCS-GNU: varargs_vec_2i
// APCS-GNU: alloca <2 x i32>, align 8
// APCS-GNU: [[VAR_ALIGN:%.*]] = alloca <2 x i32>
// APCS-GNU: [[AP_NEXT:%.*]] = getelementptr i8* {{%.*}}, i32 8
// APCS-GNU: bitcast <2 x i32>* [[VAR_ALIGN]] to i8*
// APCS-GNU: call void @llvm.memcpy
// APCS-GNU: load <2 x i32>* [[VAR_ALIGN]]
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
// CHECK: call arm_aapcscc double (i32, ...)* @varargs_vec_2i(i32 3, <2 x i32> {{%.*}})
// APCS-GNU: test_2i
// APCS-GNU: call double (i32, ...)* @varargs_vec_2i(i32 3, <2 x i32> {{%.*}})
  return varargs_vec_2i(3, *in);
}

double varargs_vec_3c(int fixed, ...) {
// CHECK: varargs_vec_3c
// CHECK: alloca <3 x i8>, align 4
// CHECK: [[AP_NEXT:%.*]] = getelementptr i8* [[AP:%.*]], i32 4
// CHECK: bitcast i8* [[AP]] to <3 x i8>*
// APCS-GNU: varargs_vec_3c
// APCS-GNU: alloca <3 x i8>, align 4
// APCS-GNU: [[AP_NEXT:%.*]] = getelementptr i8* [[AP:%.*]], i32 4
// APCS-GNU: bitcast i8* [[AP]] to <3 x i8>*
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
// CHECK: call arm_aapcscc double (i32, ...)* @varargs_vec_3c(i32 3, i32 {{%.*}})
// APCS-GNU: test_3c
// APCS-GNU: call double (i32, ...)* @varargs_vec_3c(i32 3, i32 {{%.*}})
  return varargs_vec_3c(3, *in);
}

double varargs_vec_5c(int fixed, ...) {
// CHECK: varargs_vec_5c
// CHECK: alloca <5 x i8>, align 8
// CHECK: [[ALIGN:%.*]] = and i32 {{%.*}}, -8
// CHECK: [[AP_ALIGN:%.*]] = inttoptr i32 [[ALIGN]] to i8*
// CHECK: [[AP_NEXT:%.*]] = getelementptr i8* [[AP_ALIGN]], i32 8
// CHECK: bitcast i8* [[AP_ALIGN]] to <5 x i8>*
// APCS-GNU: varargs_vec_5c
// APCS-GNU: alloca <5 x i8>, align 8
// APCS-GNU: [[VAR_ALIGN:%.*]] = alloca <5 x i8>
// APCS-GNU: [[AP_NEXT:%.*]] = getelementptr i8* {{%.*}}, i32 8
// APCS-GNU: bitcast <5 x i8>* [[VAR_ALIGN]] to i8*
// APCS-GNU: call void @llvm.memcpy
// APCS-GNU: load <5 x i8>* [[VAR_ALIGN]]
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
// CHECK: call arm_aapcscc double (i32, ...)* @varargs_vec_5c(i32 5, <2 x i32> {{%.*}})
// APCS-GNU: test_5c
// APCS-GNU: call double (i32, ...)* @varargs_vec_5c(i32 5, <2 x i32> {{%.*}})
  return varargs_vec_5c(5, *in);
}

double varargs_vec_9c(int fixed, ...) {
// CHECK: varargs_vec_9c
// CHECK: alloca <9 x i8>, align 16
// CHECK: [[VAR_ALIGN:%.*]] = alloca <9 x i8>
// CHECK: [[ALIGN:%.*]] = and i32 {{%.*}}, -8
// CHECK: [[AP_ALIGN:%.*]] = inttoptr i32 [[ALIGN]] to i8*
// CHECK: [[AP_NEXT:%.*]] = getelementptr i8* [[AP_ALIGN]], i32 16
// CHECK: bitcast <9 x i8>* [[VAR_ALIGN]] to i8*
// CHECK: call void @llvm.memcpy
// CHECK: load <9 x i8>* [[VAR_ALIGN]]
// APCS-GNU: varargs_vec_9c
// APCS-GNU: alloca <9 x i8>, align 16
// APCS-GNU: [[VAR_ALIGN:%.*]] = alloca <9 x i8>
// APCS-GNU: [[AP_NEXT:%.*]] = getelementptr i8* {{%.*}}, i32 16
// APCS-GNU: bitcast <9 x i8>* [[VAR_ALIGN]] to i8*
// APCS-GNU: call void @llvm.memcpy
// APCS-GNU: load <9 x i8>* [[VAR_ALIGN]]
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
// CHECK: call arm_aapcscc double (i32, ...)* @varargs_vec_9c(i32 9, <4 x i32> {{%.*}})
// APCS-GNU: test_9c
// APCS-GNU: call double (i32, ...)* @varargs_vec_9c(i32 9, <4 x i32> {{%.*}})
  return varargs_vec_9c(9, *in);
}

double varargs_vec_19c(int fixed, ...) {
// CHECK: varargs_vec_19c
// CHECK: [[AP_NEXT:%.*]] = getelementptr i8* [[AP:%.*]], i32 4
// CHECK: [[VAR:%.*]] = bitcast i8* [[AP]] to i8**
// CHECK: [[VAR2:%.*]] = load i8** [[VAR]]
// CHECK: bitcast i8* [[VAR2]] to <19 x i8>*
// APCS-GNU: varargs_vec_19c
// APCS-GNU: [[AP_NEXT:%.*]] = getelementptr i8* [[AP:%.*]], i32 4
// APCS-GNU: [[VAR:%.*]] = bitcast i8* [[AP]] to i8**
// APCS-GNU: [[VAR2:%.*]] = load i8** [[VAR]]
// APCS-GNU: bitcast i8* [[VAR2]] to <19 x i8>*
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
// CHECK: call arm_aapcscc double (i32, ...)* @varargs_vec_19c(i32 19, <19 x i8>* {{%.*}})
// APCS-GNU: test_19c
// APCS-GNU: call double (i32, ...)* @varargs_vec_19c(i32 19, <19 x i8>* {{%.*}})
  return varargs_vec_19c(19, *in);
}

double varargs_vec_3s(int fixed, ...) {
// CHECK: varargs_vec_3s
// CHECK: alloca <3 x i16>, align 8
// CHECK: [[ALIGN:%.*]] = and i32 {{%.*}}, -8
// CHECK: [[AP_ALIGN:%.*]] = inttoptr i32 [[ALIGN]] to i8*
// CHECK: [[AP_NEXT:%.*]] = getelementptr i8* [[AP_ALIGN]], i32 8
// CHECK: bitcast i8* [[AP_ALIGN]] to <3 x i16>*
// APCS-GNU: varargs_vec_3s
// APCS-GNU: alloca <3 x i16>, align 8
// APCS-GNU: [[VAR_ALIGN:%.*]] = alloca <3 x i16>
// APCS-GNU: [[AP_NEXT:%.*]] = getelementptr i8* {{%.*}}, i32 8
// APCS-GNU: bitcast <3 x i16>* [[VAR_ALIGN]] to i8*
// APCS-GNU: call void @llvm.memcpy
// APCS-GNU: load <3 x i16>* [[VAR_ALIGN]]
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
// CHECK: call arm_aapcscc double (i32, ...)* @varargs_vec_3s(i32 3, <2 x i32> {{%.*}})
// APCS-GNU: test_3s
// APCS-GNU: call double (i32, ...)* @varargs_vec_3s(i32 3, <2 x i32> {{%.*}})
  return varargs_vec_3s(3, *in);
}

double varargs_vec_5s(int fixed, ...) {
// CHECK: varargs_vec_5s
// CHECK: alloca <5 x i16>, align 16
// CHECK: [[VAR_ALIGN:%.*]] = alloca <5 x i16>
// CHECK: [[ALIGN:%.*]] = and i32 {{%.*}}, -8
// CHECK: [[AP_ALIGN:%.*]] = inttoptr i32 [[ALIGN]] to i8*
// CHECK: [[AP_NEXT:%.*]] = getelementptr i8* [[AP_ALIGN]], i32 16
// CHECK: bitcast <5 x i16>* [[VAR_ALIGN]] to i8*
// CHECK: call void @llvm.memcpy
// CHECK: load <5 x i16>* [[VAR_ALIGN]]
// APCS-GNU: varargs_vec_5s
// APCS-GNU: alloca <5 x i16>, align 16
// APCS-GNU: [[VAR_ALIGN:%.*]] = alloca <5 x i16>
// APCS-GNU: [[AP_NEXT:%.*]] = getelementptr i8* {{%.*}}, i32 16
// APCS-GNU: bitcast <5 x i16>* [[VAR_ALIGN]] to i8*
// APCS-GNU: call void @llvm.memcpy
// APCS-GNU: load <5 x i16>* [[VAR_ALIGN]]
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
// CHECK: call arm_aapcscc double (i32, ...)* @varargs_vec_5s(i32 5, <4 x i32> {{%.*}})
// APCS-GNU: test_5s
// APCS-GNU: call double (i32, ...)* @varargs_vec_5s(i32 5, <4 x i32> {{%.*}})
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
// CHECK: [[ALIGN:%.*]] = and i32 {{%.*}}, -8
// CHECK: [[AP_ALIGN:%.*]] = inttoptr i32 [[ALIGN]] to i8*
// CHECK: [[AP_NEXT:%.*]] = getelementptr i8* [[AP_ALIGN]], i32 16
// CHECK: bitcast i8* [[AP_ALIGN]] to %struct.StructWithVec*
// APCS-GNU: varargs_struct
// APCS-GNU: [[VAR_ALIGN:%.*]] = alloca %struct.StructWithVec
// APCS-GNU: [[AP_NEXT:%.*]] = getelementptr i8* {{%.*}}, i32 16
// APCS-GNU: bitcast %struct.StructWithVec* [[VAR_ALIGN]] to i8*
// APCS-GNU: call void @llvm.memcpy
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
// CHECK: call arm_aapcscc double (i32, ...)* @varargs_struct(i32 3, [2 x i64] {{%.*}})
// APCS-GNU: test_struct
// APCS-GNU: call double (i32, ...)* @varargs_struct(i32 3, [2 x i64] {{%.*}})
  return varargs_struct(3, *d);
}
