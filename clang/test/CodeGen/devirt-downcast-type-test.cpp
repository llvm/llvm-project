// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=c++17 -emit-llvm -o - %s | FileCheck %s
//
// Test that Clang emits llvm.type.test+llvm.assume on the object pointer at
// CK_BaseToDerived (static_cast<Derived*>) cast sites when the derived class
// is polymorphic and effectively final. This annotation allows the LLVM inliner
// (tryPromoteCall) to devirtualize virtual calls through the downcast pointer
// without requiring a visible vtable store.

struct Base {
  virtual void doFoo();
  void foo() { doFoo(); }
};

struct Derived final : Base {
  void doFoo() override;
};

// static_cast to a final polymorphic derived class: type.test must be emitted.
void f(Base *b) {
  static_cast<Derived *>(b)->foo();
}

// CHECK-LABEL: define {{.*}} @_Z1fP4Base(
// CHECK:         [[LOADED:%[0-9]+]] = load ptr, ptr %b.addr
// CHECK-NEXT:    [[TT:%[0-9]+]] = call i1 @llvm.type.test(ptr [[LOADED]], metadata !"_ZTS7Derived")
// CHECK-NEXT:    call void @llvm.assume(i1 [[TT]])

struct NonPolyBase {};
struct NonPolyDerived : NonPolyBase {};

// static_cast to a non-polymorphic derived class: no type.test should be emitted.
NonPolyDerived *g(NonPolyBase *b) {
  return static_cast<NonPolyDerived *>(b);
}

// CHECK-LABEL: define {{.*}} @_Z1gP11NonPolyBase(
// CHECK-NOT:     llvm.type.test
// CHECK:         ret ptr

struct NonFinalDerived : Base {
  void doFoo() override;
};

// static_cast to a non-final polymorphic derived class: no type.test should be
// emitted (the object could be a further-derived subclass with a different vtable).
void h(Base *b) {
  static_cast<NonFinalDerived *>(b)->foo();
}

// CHECK-LABEL: define {{.*}} @_Z1hP4Base(
// CHECK-NOT:     llvm.type.test
// CHECK:         ret void
