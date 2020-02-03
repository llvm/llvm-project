// RUN: %clang_cc1 -triple i386-unknown-unknown -O1 -emit-llvm -o - %s | FileCheck %s
// CHECK-LABEL: define i32 @f0
// CHECK:   ret i32 1
// CHECK-LABEL: define i32 @f1
// CHECK:   ret i32 1
// CHECK-LABEL: define i32 @f2
// CHECK:   ret i32 1
// <rdr://6115726>

int f0() {
  int x;
  unsigned short n = 1;
  int *a = &x;
  int *b = &x;
  a = a - n;
  b -= n;
  return a == b;
}

int f1(int *a) {
  long b = a - (int*) 1;
  a -= (int*) 1;
  return b == (long) a;
}

int f2(long n) {
  int *b = n + (int*) 1;
  n += (int*) 1;
  return b == (int*) n;
}

