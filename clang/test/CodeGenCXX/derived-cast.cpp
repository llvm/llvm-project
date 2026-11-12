// RUN: %clang_cc1 -triple x86_64-unknown-unknown -emit-llvm %s -o - | FileCheck %s

class A {
    int a;
};

class B {
    int b;
public:
    A *getAsA();
};

class X : public A, public B {
    int x;
};

// PR35909 - https://bugs.llvm.org/show_bug.cgi?id=35909

A *B::getAsA() {
  return static_cast<X*>(this);

  // CHECK-LABEL: define{{.*}} ptr @_ZN1B6getAsAEv
  // CHECK: %[[THIS:.*]] = load ptr, ptr
  // CHECK-NEXT: getelementptr inbounds i8, ptr %[[THIS]], i64 -4
}

