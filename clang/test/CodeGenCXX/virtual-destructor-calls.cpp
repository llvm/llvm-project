// RUN: %clang_cc1 -no-opaque-pointers -emit-llvm %s -o - -triple=x86_64-apple-darwin10 -mconstructor-aliases -O1 -disable-llvm-passes | FileCheck %s

struct Member {
  ~Member();
};

struct A {
  virtual ~A();
};

struct B : A {
  Member m;
  virtual ~B();
};

// Complete dtor: just an alias because there are no virtual bases.
// CHECK: @_ZN1BD1Ev ={{.*}} unnamed_addr alias {{.*}} @_ZN1BD2Ev

// (aliases from C)
// CHECK: @_ZN1CD2Ev ={{.*}} unnamed_addr alias {{.*}}, bitcast {{.*}} @_ZN1BD2Ev
// CHECK: @_ZN1CD1Ev ={{.*}} unnamed_addr alias {{.*}} @_ZN1CD2Ev

// Base dtor: actually calls A's base dtor.
// CHECK-LABEL: define{{.*}} void @_ZN1BD2Ev(%struct.B* {{[^,]*}} %this) unnamed_addr
// CHECK: call void @_ZN6MemberD1Ev
// CHECK: call void @_ZN1AD2Ev

// Deleting dtor: defers to the complete dtor.
// CHECK-LABEL: define{{.*}} void @_ZN1BD0Ev(%struct.B* {{[^,]*}} %this) unnamed_addr
// CHECK: call void @_ZN1BD1Ev
// CHECK: call void @_ZdlPv

B::~B() { }

struct C : B {
  ~C();
};

C::~C() { }

// Complete dtor: just an alias (checked above).

// Deleting dtor: defers to the complete dtor.
// CHECK-LABEL: define{{.*}} void @_ZN1CD0Ev(%struct.C* {{[^,]*}} %this) unnamed_addr
// CHECK: call void @_ZN1CD1Ev
// CHECK: call void @_ZdlPv

// Base dtor: just an alias to B's base dtor.

namespace PR12798 {
  // A qualified call to a base class destructor should not undergo virtual
  // dispatch. Template instantiation used to lose the qualifier.
  struct A { virtual ~A(); };
  template<typename T> void f(T *p) { p->A::~A(); }

  // CHECK: define {{.*}} @_ZN7PR127981fINS_1AEEEvPT_(
  // CHECK: call void @_ZN7PR127981AD1Ev(
  template void f(A*);
}
