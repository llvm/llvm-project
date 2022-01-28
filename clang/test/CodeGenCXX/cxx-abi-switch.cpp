// Assert that the ABI switch uses the proper codegen. Fuchsia uses the
// "return this" ABI on constructors and destructors by default, but if we
// explicitly choose the generic itanium C++ ABI, we should not return "this" on
// ctors/dtors.
//
// RUN: %clang_cc1 %s -emit-llvm -o - -triple=x86_64-unknown-fuchsia -fc++-abi=itanium | FileCheck %s
// RUN: %clang_cc1 %s -emit-llvm -o - -triple=aarch64-unknown-fuchsia -fc++-abi=itanium | FileCheck %s

class A {
public:
  virtual ~A();
  int x_;
};

class B : public A {
public:
  B(int *i);
  virtual ~B();
  int *i_;
};

B::B(int *i) : i_(i) {}
B::~B() {}

// CHECK: define{{.*}} void @_ZN1BC2EPi(%class.B* {{[^,]*}} %this, i32* noundef %i)
// CHECK: define{{.*}} void @_ZN1BC1EPi(%class.B* {{[^,]*}} %this, i32* noundef %i)
// CHECK: define{{.*}} void @_ZN1BD2Ev(%class.B* {{[^,]*}} %this)
// CHECK: define{{.*}} void @_ZN1BD1Ev(%class.B* {{[^,]*}} %this)
