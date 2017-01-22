// RUN: %clang_cc1 -flto=thin -flto-unit -triple x86_64-unknown-linux -fvisibility hidden -emit-llvm-bc -o %t %s
// RUN: llvm-modextract -o - -n 1 %t | llvm-dis | FileCheck %s

// CHECK: @_ZTV1A = linkonce_odr
class A {
  virtual void f() {}
};

A *f() {
  return new A;
}
