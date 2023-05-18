// RUN: %clang_cc1 %s -triple=i686-linux-gnu -emit-llvm -o - | FileCheck %s

struct S {
  mutable int n;
};
int f() {
  // The purpose of this test is to ensure that this variable is a global
  // not a constant.
  // CHECK: @_ZZ1fvE1s = internal global {{.*}} { i32 12 }
  static const S s = { 12 };
  return ++s.n;
}
