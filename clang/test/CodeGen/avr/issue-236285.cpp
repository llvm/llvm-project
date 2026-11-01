// RUN: %clang_cc1 -triple avr -mconstructor-aliases -emit-llvm -o - %s | FileCheck %s

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

// CHECK: {{.*}} alias void (ptr), ptr addrspace(1)
// CHECK: {{.*}} alias void (ptr), ptr addrspace(1)
