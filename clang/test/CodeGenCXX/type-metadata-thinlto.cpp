// RUN: %clang_cc1 -flto=thin -flto-unit -fsplit-lto-unit -triple x86_64-unknown-linux -fvisibility=hidden -emit-llvm-bc -o %t %s
// RUN: llvm-modextract -o - -n 1 %t | llvm-dis | FileCheck %s
// RUN: llvm-modextract -b -o - -n 1 %t | llvm-bcanalyzer -dump | FileCheck %s --check-prefix=LTOUNIT
// LTOUNIT: <FLAGS op0=8/>

// CHECK: @_ZTV1A = linkonce_odr
class A {
  virtual void f() {}
};

A *f() {
  return new A;
}
