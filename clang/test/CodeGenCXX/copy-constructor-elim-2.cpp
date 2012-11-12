// RUN: %clang_cc1 -triple armv7-none-eabi -emit-llvm -o - %s | FileCheck %s

struct A { int x; A(int); ~A(); };
A f() { return A(0); }
// CHECK: define void @_Z1fv
// CHECK: call {{.*}} @_ZN1AC1Ei
// CHECK-NEXT: ret void

// Verify that we do not elide copies when constructing a base class.
namespace no_elide_base {
  struct Base { 
    Base(const Base&);
    ~Base();
  };

  struct Other {
    operator Base() const;
  };

  struct Derived : public virtual Base { 
    Derived(const Other &O);
  };

  // CHECK: define {{.*}} @_ZN13no_elide_base7DerivedC1ERKNS_5OtherE(%"struct.no_elide_base::Derived"* %this, %"struct.no_elide_base::Other"* %O) unnamed_addr
  Derived::Derived(const Other &O) 
    // CHECK: call {{.*}} @_ZNK13no_elide_base5OthercvNS_4BaseEEv
    // CHECK: call {{.*}} @_ZN13no_elide_base4BaseC2ERKS0_
    // CHECK: call {{.*}} @_ZN13no_elide_base4BaseD1Ev
    : Base(O)
  {
    // CHECK: ret
  }
}

// PR8683.

namespace PR8683 {

struct A {
  A();
  A(const A&);
  A& operator=(const A&);
};

struct B {
  A a;
};

void f() {
  // Verify that we don't mark the copy constructor in this expression as elidable.
  // CHECK: call {{.*}} @_ZN6PR86831AC1ERKS0_
  A a = (B().a);
}

}

namespace PR12139 {
  struct A {
    A() : value(1) { }
    A(A const &, int value = 2) : value(value) { }
    int value;

    static A makeA() { A a; a.value = 2; return a; }
  };

  // CHECK: define i32 @_ZN7PR121394testEv
  int test() {
    // CHECK: call void @_ZN7PR121391A5makeAEv
    // CHECK-NEXT: call %"struct.PR12139::A"* @_ZN7PR121391AC1ERKS0_i
    A a(A::makeA(), 3);
    // CHECK-NEXT: getelementptr inbounds
    // CHECK-NEXT: load
    // CHECK-NEXT: ret i32
    return a.value;
  }
}

