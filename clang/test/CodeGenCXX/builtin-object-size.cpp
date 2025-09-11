// RUN: %clang_cc1 -triple x86_64-apple-darwin -fstrict-flex-arrays=0 -std=c++23 -emit-llvm -o - %s | FileCheck %s

struct EmptyS {
  int i;
  char a[];
};

template <unsigned N>
struct S {
  int i;
  char a[N];
};

// CHECK-LABEL: define noundef i32 @_Z4testRK6EmptyS(
// CHECK: ret i32 0
unsigned test(const EmptyS &empty) {
  return __builtin_object_size(empty.a, 3);
}

// CHECK-LABEL: define noundef i32 @_Z4testRK1SILj2EE(
// CHECK: ret i32 0
unsigned test(const S<2> &s2) {
  return __builtin_object_size(s2.a, 3);
}

// CHECK-LABEL: define noundef i32 @_Z4testRi(
// CHECK: ret i32 0
unsigned test(int &i) {
  return __builtin_object_size(&i, 3);
}
