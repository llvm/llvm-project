// RUN: %clang_cc1 -emit-llvm-only -triple %itanium_abi_triple %s -emit-llvm -o - %s | FileCheck %s

class Base1 {
  virtual void Foo1();
};

class Base2 {
  virtual void Foo2();
};

class alignas(16) Obj : public Base1, public Base2 {
  void Foo1() override;
  void Foo2() override;
  ~Obj();
};

void Obj::Foo1() {}
void Obj::Foo2() {}

// CHECK: define dso_local void @_ZN3Obj4Foo2Ev(%class.Obj.0* nonnull dereferenceable(16) %this) unnamed_addr #0 align 2 {

// FIXME: the argument should be  %class.Base2.2* nonnull dereferenceable(8) %this
// CHECK: define dso_local void @_ZThn8_N3Obj4Foo2Ev(%class.Obj.0* %this) unnamed_addr #1 align 2 {

// CHECK: tail call void @_ZN3Obj4Foo2Ev(%class.Obj.0* nonnull dereferenceable(16) %2)
