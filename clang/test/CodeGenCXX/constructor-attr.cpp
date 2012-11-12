// RUN: %clang_cc1 -emit-llvm -o - %s | FileCheck %s

// CHECK: @llvm.global_ctors

// PR6521
void bar();
struct Foo {
  // CHECK: define linkonce_odr void @_ZN3Foo3fooEv
  static void foo() __attribute__((constructor)) {
    bar();
  }
};
