// RUN: %clang_cc1 -triple aarch64-windows -emit-llvm -w -o - %s | FileCheck %s

// CHECK: target datalayout = "e-m:w-p:64:64-i32:32-i64:64-i128:128-n32:64-S128-Fn32"
// CHECK: target triple = "aarch64-unknown-windows-msvc"

int check_short(void) {
  return sizeof(short);
// CHECK: ret i32 2
}

int check_int(void) {
  return sizeof(int);
// CHECK: ret i32 4
}

int check_long(void) {
  return sizeof(long);
// CHECK: ret i32 4
}

int check_longlong(void) {
  return sizeof(long long);
// CHECK: ret i32 8
}

int check_int128(void) {
  return sizeof(__int128);
// CHECK: ret i32 16
}

int check_fp16(void) {
  return sizeof(__fp16);
// CHECK: ret i32 2
}

int check_float(void) {
  return sizeof(float);
// CHECK: ret i32 4
}

int check_double(void) {
  return sizeof(double);
// CHECK: ret i32 8
}

int check_longdouble(void) {
  return sizeof(long double);
// CHECK: ret i32 8
}

int check_floatComplex(void) {
  return sizeof(float _Complex);
// CHECK: ret i32 8
}

int check_doubleComplex(void) {
  return sizeof(double _Complex);
// CHECK: ret i32 16
}

int check_longdoubleComplex(void) {
  return sizeof(long double _Complex);
// CHECK: ret i32 16
}

int check_bool(void) {
  return sizeof(_Bool);
// CHECK: ret i32 1
}

int check_wchar(void) {
  return sizeof(__WCHAR_TYPE__);
// CHECK: ret i32 2
}

int check_wchar_unsigned(void) {
  return (__WCHAR_TYPE__)-1 > (__WCHAR_TYPE__)0;
// CHECK: ret i32 1
}

enum Small {
  Item
};

int foo(void) {
  return sizeof(enum Small);
// CHECK: ret i32 4
}
