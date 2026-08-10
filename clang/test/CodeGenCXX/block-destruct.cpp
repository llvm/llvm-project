// RUN: %clang_cc1 %s -fblocks -triple x86_64-apple-darwin -emit-llvm -o - | FileCheck %s

struct A { ~A(); };

void f() {
  __block A a;
}

// CHECK: call void @_ZN1AD1Ev
