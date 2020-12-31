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

  // CHECK-LABEL: define{{.*}} %class.A* @_ZN1B6getAsAEv
  // CHECK: %[[THIS:.*]] = load %class.B*, %class.B**
  // CHECK-NEXT: %[[BC:.*]] = bitcast %class.B* %[[THIS]] to i8*
  // CHECK-NEXT: getelementptr inbounds i8, i8* %[[BC]], i64 -4
}

