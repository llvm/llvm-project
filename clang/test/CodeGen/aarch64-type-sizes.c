// RUN: %clang_cc1 -triple aarch64_be-none-linux-gnu -emit-llvm -w -o - %s | FileCheck %s
// char by definition has size 1

// CHECK: target datalayout = "E-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128"

int check_short() {
  return sizeof(short);
// CHECK: ret i32 2
}

int check_int() {
  return sizeof(int);
// CHECK: ret i32 4
}

int check_long() {
// Both 4 and 8 are permitted under the PCS, Linux says 8!
  return sizeof(long);
// CHECK: ret i32 8
}

int check_longlong() {
  return sizeof(long long);
// CHECK: ret i32 8
}

int check_int128() {
  return sizeof(__int128);
// CHECK: ret i32 16
}

int check_fp16() {
  return sizeof(__fp16);
// CHECK: ret i32 2
}

int check_float() {
  return sizeof(float);
// CHECK: ret i32 4
}

int check_double() {
  return sizeof(double);
// CHECK: ret i32 8
}

int check_longdouble() {
  return sizeof(long double);
// CHECK: ret i32 16
}

int check_floatComplex() {
  return sizeof(float _Complex);
// CHECK: ret i32 8
}

int check_doubleComplex() {
  return sizeof(double _Complex);
// CHECK: ret i32 16
}

int check_longdoubleComplex() {
  return sizeof(long double _Complex);
// CHECK: ret i32 32
}

int check_bool() {
  return sizeof(_Bool);
// CHECK: ret i32 1
}

int check_wchar() {
// PCS allows either unsigned short or unsigned int. Linux again says "bigger!"
  return sizeof(__WCHAR_TYPE__);
// CHECK: ret i32 4
}

int check_wchar_unsigned() {
  return (__WCHAR_TYPE__)-1 > (__WCHAR_TYPE__)0;
// CHECK: ret i32 1
}

enum Small {
  Item
};

int foo() {
  return sizeof(enum Small);
// CHECK: ret i32 4
}
