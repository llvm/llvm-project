// Cross comdat example
// Parent VTable is in a comdat section.

// RUN: %clang_cc1 %s -triple=aarch64 -S -o - -emit-llvm -fexperimental-relative-c++-abi-vtables | FileCheck %s
// RUN: %clang_cc1 %s -triple=x86_64 -S -o - -emit-llvm -fexperimental-relative-c++-abi-vtables | FileCheck %s

// A::foo() has a comdat since it is an inline function
// CHECK: $_ZN1A3fooEv = comdat any
// CHECK: $_ZTV1A = comdat any
// CHECK: $_ZTS1A = comdat any

// The VTable for A has its own comdat section bc it has no key function
// CHECK: $_ZTI1A = comdat any
// CHECK: $_ZTI1A.rtti_proxy = comdat any

// The VTable for A is emitted here and in a comdat section since it has no key function, and is used in this module when creating an instance of A.
// CHECK: @_ZTV1A.local = linkonce_odr hidden unnamed_addr constant { [3 x i32] } { [3 x i32] [i32 0, i32 trunc (i64 sub (i64 ptrtoint (ptr @_ZTI1A.rtti_proxy to i64), i64 ptrtoint (ptr getelementptr inbounds ({ [3 x i32] }, ptr @_ZTV1A.local, i32 0, i32 0, i32 2) to i64)) to i32), i32 trunc (i64 sub (i64 ptrtoint (ptr dso_local_equivalent @_ZN1A3fooEv to i64), i64 ptrtoint (ptr getelementptr inbounds ({ [3 x i32] }, ptr @_ZTV1A.local, i32 0, i32 0, i32 2) to i64)) to i32)] }, comdat($_ZTV1A), align 4
// CHECK: @_ZTVN10__cxxabiv117__class_type_infoE = external global [0 x ptr]
// CHECK: @_ZTS1A = linkonce_odr constant [3 x i8] c"1A\00", comdat, align 1
// CHECK: @_ZTI1A = linkonce_odr constant { ptr, ptr } { ptr getelementptr inbounds (i8, ptr @_ZTVN10__cxxabiv117__class_type_infoE, i32 8), ptr @_ZTS1A }, comdat, align 8
// CHECK: @_ZTI1A.rtti_proxy = hidden unnamed_addr constant ptr @_ZTI1A, comdat
// CHECK: @_ZTV1A = linkonce_odr unnamed_addr alias { [3 x i32] }, ptr @_ZTV1A.local

// CHECK:      define linkonce_odr void @_ZN1A3fooEv(ptr {{.*}}%this) unnamed_addr #{{[0-9]+}} comdat

class A {
public:
  inline virtual void foo() {}
};
class B : public A {
public:
  void foo() override;
};
void A_foo(A *a);

void A_foo(A *a) { a->foo(); }

// func() is only used to emit a vtable.
void func() {
  A a;
  A_foo(&a);
}
