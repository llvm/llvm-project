// RUN: %clang_cc1 -triple avr -target-cpu atmega328 -emit-llvm -x c++ < %s | FileCheck --allow-empty %s

// This test case verifies https://github.com/llvm/llvm-project/issues/236285 .

struct A {};
struct B : A {
  ~B();
};

B::~B() {}

struct C : B {
  ~C();
};

C::~C() {}

// CHECK: {{.*}} alias void (ptr), ptr
