// RUN: %clang_cc1 -emit-llvm %s -o - -triple=x86_64-apple-darwin10 | FileCheck %s

// PR5392
namespace PR5392 {
struct A
{
  static int a;
};

A a1;
void f()
{
  // CHECK: store i32 10, i32* @_ZN6PR53921A1aE
  a1.a = 10;
  // CHECK: store i32 20, i32* @_ZN6PR53921A1aE
  A().a = 20;
}

}

struct A {
  A();
  ~A();
  enum E { Foo };
};

A *g();

void f(A *a) {
  A::E e1 = a->Foo;
  
  // CHECK: call %struct.A* @_Z1gv()
  A::E e2 = g()->Foo;
  // CHECK: call void @_ZN1AC1Ev(
  // CHECK: call void @_ZN1AD1Ev(
  A::E e3 = A().Foo;
}

namespace test3 {
struct A {
  static int foo();
};
int f() {
  return A().foo();
}
}

namespace test4 {
  struct A {
    int x;
  };
  struct B {
    int x;
    void foo();
  };
  struct C : A, B {
  };

  extern C *c_ptr;

  // CHECK: define i32 @_ZN5test44testEv()
  int test() {
    // CHECK: load {{.*}} @_ZN5test45c_ptrE
    // CHECK-NEXT: bitcast
    // CHECK-NEXT: getelementptr
    // CHECK-NEXT: bitcast
    // CHECK-NEXT: call void @_ZN5test41B3fooEv
    c_ptr->B::foo();

    // CHECK: load {{.*}} @_ZN5test45c_ptrE
    // CHECK-NEXT: bitcast
    // CHECK-NEXT: getelementptr
    // CHECK-NEXT: bitcast
    // CHECK-NEXT: getelementptr
    // CHECK-NEXT: store i32 5
    c_ptr->B::x = 5;

    // CHECK: load {{.*}} @_ZN5test45c_ptrE
    // CHECK-NEXT: bitcast
    // CHECK-NEXT: getelementptr
    // CHECK-NEXT: bitcast
    // CHECK-NEXT: getelementptr
    // CHECK-NEXT: load i32*
    return c_ptr->B::x;
  }
}
