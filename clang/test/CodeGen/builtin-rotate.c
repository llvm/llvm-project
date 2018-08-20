// RUN: %clang_cc1 %s -emit-llvm -o - | FileCheck %s

unsigned char rotl8(unsigned char x, unsigned char y) {
// CHECK-LABEL: rotl8
// CHECK: [[F:%.*]] = call i8 @llvm.fshl.i8(i8 [[X:%.*]], i8 [[X]], i8 [[Y:%.*]])
// CHECK-NEXT: ret i8 [[F]]

  return __builtin_rotateleft8(x, y);
}

short rotl16(short x, short y) {
// CHECK-LABEL: rotl16
// CHECK: [[F:%.*]] = call i16 @llvm.fshl.i16(i16 [[X:%.*]], i16 [[X]], i16 [[Y:%.*]])
// CHECK-NEXT: ret i16 [[F]]

  return __builtin_rotateleft16(x, y);
}

int rotl32(int x, unsigned int y) {
// CHECK-LABEL: rotl32
// CHECK: [[F:%.*]] = call i32 @llvm.fshl.i32(i32 [[X:%.*]], i32 [[X]], i32 [[Y:%.*]])
// CHECK-NEXT: ret i32 [[F]]

  return __builtin_rotateleft32(x, y);
}

unsigned long long rotl64(unsigned long long x, long long y) {
// CHECK-LABEL: rotl64
// CHECK: [[F:%.*]] = call i64 @llvm.fshl.i64(i64 [[X:%.*]], i64 [[X]], i64 [[Y:%.*]])
// CHECK-NEXT: ret i64 [[F]]

  return __builtin_rotateleft64(x, y);
}

char rotr8(char x, char y) {
// CHECK-LABEL: rotr8
// CHECK: [[F:%.*]] = call i8 @llvm.fshr.i8(i8 [[X:%.*]], i8 [[X]], i8 [[Y:%.*]])
// CHECK-NEXT: ret i8 [[F]]

  return __builtin_rotateright8(x, y);
}

unsigned short rotr16(unsigned short x, unsigned short y) {
// CHECK-LABEL: rotr16
// CHECK: [[F:%.*]] = call i16 @llvm.fshr.i16(i16 [[X:%.*]], i16 [[X]], i16 [[Y:%.*]])
// CHECK-NEXT: ret i16 [[F]]

  return __builtin_rotateright16(x, y);
}

unsigned int rotr32(unsigned int x, int y) {
// CHECK-LABEL: rotr32
// CHECK: [[F:%.*]] = call i32 @llvm.fshr.i32(i32 [[X:%.*]], i32 [[X]], i32 [[Y:%.*]])
// CHECK-NEXT: ret i32 [[F]]

  return __builtin_rotateright32(x, y);
}

long long rotr64(long long x, unsigned long long y) {
// CHECK-LABEL: rotr64
// CHECK: [[F:%.*]] = call i64 @llvm.fshr.i64(i64 [[X:%.*]], i64 [[X]], i64 [[Y:%.*]])
// CHECK-NEXT: ret i64 [[F]]

  return __builtin_rotateright64(x, y);
}

