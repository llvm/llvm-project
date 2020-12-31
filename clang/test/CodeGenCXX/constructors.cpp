// RUN: %clang_cc1 -triple x86_64-apple-darwin10 %s -emit-llvm -o - | FileCheck %s --implicit-check-not=should_not_appear_in_output --check-prefixes=CHECK,NULL-INVALID
// RUN: %clang_cc1 -triple x86_64-apple-darwin10 %s -emit-llvm -fno-delete-null-pointer-checks -o - | FileCheck %s --implicit-check-not=should_not_appear_in_output --check-prefixes=CHECK,NULL-VALID

struct Member { int x; Member(); Member(int); Member(const Member &); };
struct VBase { int x; VBase(); VBase(int); VBase(const VBase &); };

struct ValueClass {
  ValueClass(int x, int y) : x(x), y(y) {}
  int x;
  int y;
}; // subject to ABI trickery



/* Test basic functionality. */
struct A {
  A(struct Undeclared &);
  A(ValueClass);
  Member mem;
};

A::A(struct Undeclared &ref) : mem(0) {}

// Check that delegation works.
// NULL-INVALID-LABEL: define{{.*}} void @_ZN1AC2ER10Undeclared(%struct.A* {{[^,]*}} %this, %struct.Undeclared* nonnull align 1 %ref) unnamed_addr
// NULL-VALID-LABEL: define{{.*}} void @_ZN1AC2ER10Undeclared(%struct.A* {{[^,]*}} %this, %struct.Undeclared* align 1 %ref) unnamed_addr
// CHECK: call void @_ZN6MemberC1Ei(

// NULL-INVALID-LABEL: define{{.*}} void @_ZN1AC1ER10Undeclared(%struct.A* {{[^,]*}} %this, %struct.Undeclared* nonnull align 1 %ref) unnamed_addr
// NULL-VALID-LABEL: define{{.*}} void @_ZN1AC1ER10Undeclared(%struct.A* {{[^,]*}} %this, %struct.Undeclared* align 1 %ref) unnamed_addr
// CHECK: call void @_ZN1AC2ER10Undeclared(

A::A(ValueClass v) : mem(v.y - v.x) {}

// CHECK-LABEL: define{{.*}} void @_ZN1AC2E10ValueClass(%struct.A* {{[^,]*}} %this, i64 %v.coerce) unnamed_addr
// CHECK: call void @_ZN6MemberC1Ei(

// CHECK-LABEL: define{{.*}} void @_ZN1AC1E10ValueClass(%struct.A* {{[^,]*}} %this, i64 %v.coerce) unnamed_addr
// CHECK: call void @_ZN1AC2E10ValueClass(

/* Test that things work for inheritance. */
struct B : A {
  B(struct Undeclared &);
  Member mem;
};

B::B(struct Undeclared &ref) : A(ref), mem(1) {}

// NULL-INVALID-LABEL: define{{.*}} void @_ZN1BC2ER10Undeclared(%struct.B* {{[^,]*}} %this, %struct.Undeclared* nonnull align 1 %ref) unnamed_addr
// NULL-VALID-LABEL: define{{.*}} void @_ZN1BC2ER10Undeclared(%struct.B* {{[^,]*}} %this, %struct.Undeclared* align 1 %ref) unnamed_addr
// CHECK: call void @_ZN1AC2ER10Undeclared(
// CHECK: call void @_ZN6MemberC1Ei(

// NULL-INVALID-LABEL: define{{.*}} void @_ZN1BC1ER10Undeclared(%struct.B* {{[^,]*}} %this, %struct.Undeclared* nonnull align 1 %ref) unnamed_addr
// NULL-VALID-LABEL: define{{.*}} void @_ZN1BC1ER10Undeclared(%struct.B* {{[^,]*}} %this, %struct.Undeclared* align 1 %ref) unnamed_addr
// CHECK: call void @_ZN1BC2ER10Undeclared(


/* Test that the delegation optimization is disabled for classes with
   virtual bases (for now).  This is necessary because a vbase
   initializer could access one of the parameter variables by
   reference.  That's a solvable problem, but let's not solve it right
   now. */
struct C : virtual A {
  C(int);
  Member mem;
};
C::C(int x) : A(ValueClass(x, x+1)), mem(x * x) {}

// CHECK-LABEL: define{{.*}} void @_ZN1CC2Ei(%struct.C* {{[^,]*}} %this, i8** %vtt, i32 %x) unnamed_addr
// CHECK: call void @_ZN6MemberC1Ei(

// CHECK-LABEL: define{{.*}} void @_ZN1CC1Ei(%struct.C* {{[^,]*}} %this, i32 %x) unnamed_addr
// CHECK: call void @_ZN10ValueClassC1Eii(
// CHECK: call void @_ZN1AC2E10ValueClass(
// CHECK: call void @_ZN6MemberC1Ei(


/* Test that the delegation optimization is disabled for varargs
   constructors. */
struct D : A {
  D(int, ...);
  Member mem;
};

D::D(int x, ...) : A(ValueClass(x, x+1)), mem(x*x) {}

// CHECK-LABEL: define{{.*}} void @_ZN1DC2Eiz(%struct.D* {{[^,]*}} %this, i32 %x, ...) unnamed_addr
// CHECK: call void @_ZN10ValueClassC1Eii(
// CHECK: call void @_ZN1AC2E10ValueClass(
// CHECK: call void @_ZN6MemberC1Ei(

// CHECK-LABEL: define{{.*}} void @_ZN1DC1Eiz(%struct.D* {{[^,]*}} %this, i32 %x, ...) unnamed_addr
// CHECK: call void @_ZN10ValueClassC1Eii(
// CHECK: call void @_ZN1AC2E10ValueClass(
// CHECK: call void @_ZN6MemberC1Ei(

// PR6622:  this shouldn't crash
namespace test0 {
  struct A {};
  struct B : virtual A { int x; };
  struct C : B {};
  
  void test(C &in) {
    C tmp = in;
  }
}

namespace test1 {
  struct A { A(); void *ptr; };
  struct B { B(); int x; A a[0]; };
  B::B() {}
  // CHECK-LABEL:    define{{.*}} void @_ZN5test11BC2Ev(
  // CHECK:      [[THIS:%.*]] = load [[B:%.*]]*, [[B:%.*]]**
  // CHECK-NEXT: ret void
}

// Ensure that we
// a) emit the ABI-required but useless complete object and deleting destructor
//    symbols for an abstract class, and 
// b) do *not* emit references to virtual base destructors for an abstract class
//
// Our approach to this is to give these functions a body that simply traps.
//
// FIXME: We should ideally not create these symbols at all, but Clang can
// actually generate references to them in other TUs in some cases, so we can't
// stop emitting them without breaking ABI. See:
//
//   https://github.com/itanium-cxx-abi/cxx-abi/issues/10
namespace abstract {
  // Note, the destructor of this class is not instantiated here.
  template<typename T> struct should_not_appear_in_output {
    ~should_not_appear_in_output() { int arr[-(int)sizeof(T)]; }
  };

  struct X { ~X(); };

  struct A : virtual should_not_appear_in_output<int>, X {
    virtual ~A() = 0;
  };

  // CHECK-LABEL: define{{.*}} void @_ZN8abstract1AD2Ev(
  // CHECK: call {{.*}}@_ZN8abstract1XD2Ev(
  // CHECK: ret

  // CHECK-LABEL: define{{.*}} void @_ZN8abstract1AD1Ev(
  // CHECK: call {{.*}}@llvm.trap(
  // CHECK: unreachable

  // CHECK-LABEL: define{{.*}} void @_ZN8abstract1AD0Ev(
  // CHECK: call {{.*}}@llvm.trap(
  // CHECK: unreachable
  A::~A() {}

  struct B : virtual should_not_appear_in_output<int>, X {
    virtual void f() = 0;
    ~B();
  };

  // CHECK-LABEL: define{{.*}} void @_ZN8abstract1BD2Ev(
  // CHECK: call {{.*}}@_ZN8abstract1XD2Ev(
  // CHECK: ret

  // CHECK-LABEL: define{{.*}} void @_ZN8abstract1BD1Ev(
  // CHECK: call {{.*}}@llvm.trap(
  // CHECK: unreachable

  // CHECK-NOT: @_ZN8abstract1BD0Ev(
  B::~B() {}
}
