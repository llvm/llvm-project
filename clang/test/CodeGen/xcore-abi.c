// REQUIRES: xcore-registered-target
// RUN: %clang_cc1 -no-opaque-pointers -triple xcore -verify %s
_Static_assert(sizeof(long long) == 8, "sizeof long long is wrong");
_Static_assert(_Alignof(long long) == 4, "alignof long long is wrong");

_Static_assert(sizeof(double) == 8, "sizeof double is wrong");
_Static_assert(_Alignof(double) == 4, "alignof double is wrong");

// RUN: %clang_cc1 -no-opaque-pointers -triple xcore-unknown-unknown -fno-signed-char -fno-common -emit-llvm -o - %s | FileCheck %s

// CHECK: target triple = "xcore-unknown-unknown"

// CHECK: @cgx = external constant i32, section ".cp.rodata"
extern const int cgx;
int fcgx() { return cgx;}
// CHECK: @g1 ={{.*}} global i32 0, align 4
int g1;
// CHECK: @cg1 ={{.*}} constant i32 0, section ".cp.rodata", align 4
const int cg1;

#include <stdarg.h>
struct x { int a[5]; };
void f(void*);
void testva (int n, ...) {
  // CHECK-LABEL: testva
  va_list ap;
  va_start(ap,n);
  // CHECK: [[AP:%[a-z0-9]+]] = alloca i8*, align 4
  // CHECK: [[AP1:%[a-z0-9]+]] = bitcast i8** [[AP]] to i8*
  // CHECK: call void @llvm.va_start(i8* [[AP1]])

  char* v1 = va_arg (ap, char*);
  f(v1);
  // CHECK: [[I:%[a-z0-9]+]] = load i8*, i8** [[AP]]
  // CHECK: [[P:%[a-z0-9]+]] = bitcast i8* [[I]] to i8**
  // CHECK: [[IN:%[a-z0-9]+]] = getelementptr inbounds i8, i8* [[I]], i32 4
  // CHECK: store i8* [[IN]], i8** [[AP]]
  // CHECK: [[V1:%[a-z0-9]+]] = load i8*, i8** [[P]]
  // CHECK: store i8* [[V1]], i8** [[V:%[a-z0-9]+]], align 4
  // CHECK: [[V2:%[a-z0-9]+]] = load i8*, i8** [[V]], align 4
  // CHECK: call void @f(i8* noundef [[V2]])

  char v2 = va_arg (ap, char); // expected-warning{{second argument to 'va_arg' is of promotable type 'char'}}
  f(&v2);
  // CHECK: [[I:%[a-z0-9]+]] = load i8*, i8** [[AP]]
  // CHECK: [[IN:%[a-z0-9]+]] = getelementptr inbounds i8, i8* [[I]], i32 4
  // CHECK: store i8* [[IN]], i8** [[AP]]
  // CHECK: [[V1:%[a-z0-9]+]] = load i8, i8* [[I]]
  // CHECK: store i8 [[V1]], i8* [[V:%[a-z0-9]+]], align 1
  // CHECK: call void @f(i8* noundef [[V]])

  int v3 = va_arg (ap, int);
  f(&v3);
  // CHECK: [[I:%[a-z0-9]+]] = load i8*, i8** [[AP]]
  // CHECK: [[P:%[a-z0-9]+]] = bitcast i8* [[I]] to i32*
  // CHECK: [[IN:%[a-z0-9]+]] = getelementptr inbounds i8, i8* [[I]], i32 4
  // CHECK: store i8* [[IN]], i8** [[AP]]
  // CHECK: [[V1:%[a-z0-9]+]] = load i32, i32* [[P]]
  // CHECK: store i32 [[V1]], i32* [[V:%[a-z0-9]+]], align 4
  // CHECK: [[V2:%[a-z0-9]+]] = bitcast i32* [[V]] to i8*
  // CHECK: call void @f(i8* noundef [[V2]])

  long long int v4 = va_arg (ap, long long int);
  f(&v4);
  // CHECK: [[I:%[a-z0-9]+]] = load i8*, i8** [[AP]]
  // CHECK: [[P:%[a-z0-9]+]] = bitcast i8* [[I]] to i64*
  // CHECK: [[IN:%[a-z0-9]+]] = getelementptr inbounds i8, i8* [[I]], i32 8
  // CHECK: store i8* [[IN]], i8** [[AP]]
  // CHECK: [[V1:%[a-z0-9]+]] = load i64, i64* [[P]]
  // CHECK: store i64 [[V1]], i64* [[V:%[a-z0-9]+]], align 4
  // CHECK:[[V2:%[a-z0-9]+]] = bitcast i64* [[V]] to i8*
  // CHECK: call void @f(i8* noundef [[V2]])

  struct x v5 = va_arg (ap, struct x);  // typical aggregate type
  f(&v5);
  // CHECK: [[I:%[a-z0-9]+]] = load i8*, i8** [[AP]]
  // CHECK: [[I2:%[a-z0-9]+]] = bitcast i8* [[I]] to %struct.x**
  // CHECK: [[P:%[a-z0-9]+]] = load %struct.x*, %struct.x** [[I2]]
  // CHECK: [[IN:%[a-z0-9]+]] = getelementptr inbounds i8, i8* [[I]], i32 4
  // CHECK: store i8* [[IN]], i8** [[AP]]
  // CHECK: [[V1:%[a-z0-9]+]] = bitcast %struct.x* [[V:%[a-z0-9]+]] to i8*
  // CHECK: [[P1:%[a-z0-9]+]] = bitcast %struct.x* [[P]] to i8*
  // CHECK: call void @llvm.memcpy.p0i8.p0i8.i32(i8* align 4 [[V1]], i8* align 4 [[P1]], i32 20, i1 false)
  // CHECK: [[V2:%[a-z0-9]+]] = bitcast %struct.x* [[V]] to i8*
  // CHECK: call void @f(i8* noundef [[V2]])

  int* v6 = va_arg (ap, int[4]);  // an unusual aggregate type
  f(v6);
  // CHECK: [[I:%[a-z0-9]+]] = load i8*, i8** [[AP]]
  // CHECK: [[I2:%[a-z0-9]+]] = bitcast i8* [[I]] to [4 x i32]**
  // CHECK: [[P:%[a-z0-9]+]] = load [4 x i32]*, [4 x i32]** [[I2]]
  // CHECK: [[IN:%[a-z0-9]+]] = getelementptr inbounds i8, i8* [[I]], i32 4
  // CHECK: store i8* [[IN]], i8** [[AP]]
  // CHECK: [[V1:%[a-z0-9]+]] = bitcast [4 x i32]* [[V0:%[a-z0-9]+]] to i8*
  // CHECK: [[P1:%[a-z0-9]+]] = bitcast [4 x i32]* [[P]] to i8*
  // CHECK: call void @llvm.memcpy.p0i8.p0i8.i32(i8* align 4 [[V1]], i8* align 4 [[P1]], i32 16, i1 false)
  // CHECK: [[V2:%[a-z0-9]+]] = getelementptr inbounds [4 x i32], [4 x i32]* [[V0]], i32 0, i32 0
  // CHECK: store i32* [[V2]], i32** [[V:%[a-z0-9]+]], align 4
  // CHECK: [[V3:%[a-z0-9]+]] = load i32*, i32** [[V]], align 4
  // CHECK: [[V4:%[a-z0-9]+]] = bitcast i32* [[V3]] to i8*
  // CHECK: call void @f(i8* noundef [[V4]])

  double v7 = va_arg (ap, double);
  f(&v7);
  // CHECK: [[I:%[a-z0-9]+]] = load i8*, i8** [[AP]]
  // CHECK: [[P:%[a-z0-9]+]] = bitcast i8* [[I]] to double*
  // CHECK: [[IN:%[a-z0-9]+]] = getelementptr inbounds i8, i8* [[I]], i32 8
  // CHECK: store i8* [[IN]], i8** [[AP]]
  // CHECK: [[V1:%[a-z0-9]+]] = load double, double* [[P]]
  // CHECK: store double [[V1]], double* [[V:%[a-z0-9]+]], align 4
  // CHECK: [[V2:%[a-z0-9]+]] = bitcast double* [[V]] to i8*
  // CHECK: call void @f(i8* noundef [[V2]])
}

void testbuiltin (void) {
  // CHECK-LABEL: testbuiltin
  // CHECK: call i32 @llvm.xcore.getid()
  // CHECK: call i32 @llvm.xcore.getps(i32 {{%[a-z0-9]+}})
  // CHECK: call i32 @llvm.xcore.bitrev(i32 {{%[a-z0-9]+}})
  // CHECK: call void @llvm.xcore.setps(i32 {{%[a-z0-9]+}}, i32 {{%[a-z0-9]+}})
  volatile int i = __builtin_getid();
  volatile unsigned int ui = __builtin_getps(i);
  ui = __builtin_bitrev(ui);
  __builtin_setps(i,ui);

  // CHECK: store volatile i32 0, i32* {{%[a-z0-9]+}}, align 4
  // CHECK: store volatile i32 1, i32* {{%[a-z0-9]+}}, align 4
  // CHECK: store volatile i32 -1, i32* {{%[a-z0-9]+}}, align 4
  volatile int res;
  res = __builtin_eh_return_data_regno(0);
  res = __builtin_eh_return_data_regno(1);
  res = __builtin_eh_return_data_regno(2);
}

// CHECK-LABEL: define{{.*}} zeroext i8 @testchar()
// CHECK: ret i8 -1
char testchar (void) {
  return (char)-1;
}

// CHECK: "frame-pointer"="none"
