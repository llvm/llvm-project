// RUN: %clang_cc1 -triple arm64_32-apple-ios7.0 -target-abi darwinpcs -emit-llvm -o - -O1 -ffreestanding %s | FileCheck %s

#include <stdarg.h>

typedef struct {
  int a;
} OneInt;

// No realignment should be needed here: slot size is 4 bytes.
int test_int(OneInt input, va_list *mylist) {
// CHECK-LABEL: define{{.*}} i32 @test_int(i32 %input
// CHECK: [[START:%.*]] = load ptr, ptr %mylist
// CHECK: [[NEXT:%.*]] = getelementptr inbounds nuw i8, ptr [[START]], i32 4
// CHECK: store ptr [[NEXT]], ptr %mylist

// CHECK: [[RES:%.*]] = load i32, ptr [[START]]
// CHECK: ret i32 [[RES]]

  return va_arg(*mylist, OneInt).a;
}


typedef struct {
  long long a;
} OneLongLong;

// Minimum slot size is 4 bytes, so address needs rounding up to multiple of 8.
long long test_longlong(OneLongLong input, va_list *mylist) {
  // CHECK-LABEL: define{{.*}} i64 @test_longlong(i64 %input
  // CHECK: [[STARTPTR:%.*]] = load ptr, ptr %mylist
  // CHECK: [[ALIGN_TMP:%.+]] = getelementptr inbounds nuw i8, ptr [[STARTPTR]], i32 7
  // CHECK: [[ALIGNED_ADDR:%.+]] = tail call align 8 ptr @llvm.ptrmask.p0.i32(ptr nonnull [[ALIGN_TMP]], i32 -8)
  // CHECK: [[NEXT:%.*]] = getelementptr inbounds nuw i8, ptr [[ALIGNED_ADDR]], i32 8
  // CHECK: store ptr [[NEXT]], ptr %mylist

  // CHECK: [[RES:%.*]] = load i64, ptr [[ALIGNED_ADDR]]
  // CHECK: ret i64 [[RES]]

  return va_arg(*mylist, OneLongLong).a;
}


typedef struct {
  float arr[4];
} HFA;

// HFAs take priority over passing large structs indirectly.
float test_hfa(va_list *mylist) {
// CHECK-LABEL: define{{.*}} float @test_hfa
// CHECK: [[START:%.*]] = load ptr, ptr %mylist

// CHECK: [[NEXT:%.*]] = getelementptr inbounds nuw i8, ptr [[START]], i32 16
// CHECK: store ptr [[NEXT]], ptr %mylist

// CHECK: [[RES:%.*]] = load float, ptr [[START]]
// CHECK: ret float [[RES]]

  return va_arg(*mylist, HFA).arr[0];
}

// armv7k does not return HFAs normally for variadic functions, so we must match
// that.
HFA test_hfa_return(int n, ...) {
// CHECK-LABEL: define{{.*}} [2 x i64] @test_hfa_return
  HFA h = {0};
  return h;
}

typedef struct {
  long long a, b;
  char c;
} BigStruct;

// Structs bigger than 16 bytes are passed indirectly: a pointer is placed on
// the stack.
long long test_bigstruct(BigStruct input, va_list *mylist) {
// CHECK-LABEL: define{{.*}} i64 @test_bigstruct(ptr
// CHECK: [[START:%.*]] = load ptr, ptr %mylist
// CHECK: [[NEXT:%.*]] = getelementptr inbounds nuw i8, ptr [[START]], i32 4
// CHECK: store ptr [[NEXT]], ptr %mylist

// CHECK: [[ADDR:%.*]] = load ptr, ptr [[START]]
// CHECK: [[RES:%.*]] = load i64, ptr [[ADDR]]
// CHECK: ret i64 [[RES]]

  return va_arg(*mylist, BigStruct).a;
}

typedef struct {
  short arr[3];
} ThreeShorts;

// Slot sizes are 4-bytes on arm64_32, so structs with less than 32-bit
// alignment must be passed via "[N x i32]" to be correctly allocated in the
// backend.
short test_threeshorts(ThreeShorts input, va_list *mylist) {
// CHECK-LABEL: define{{.*}} signext i16 @test_threeshorts([2 x i32] %input

// CHECK: [[START:%.*]] = load ptr, ptr %mylist
// CHECK: [[NEXT:%.*]] = getelementptr inbounds nuw i8, ptr [[START]], i32 8
// CHECK: store ptr [[NEXT]], ptr %mylist

// CHECK: [[RES:%.*]] = load i16, ptr [[START]]
// CHECK: ret i16 [[RES]]

  return va_arg(*mylist, ThreeShorts).arr[0];
}
