// RUN: %clang_cc1 -std=c++11 %s -emit-llvm -o - -triple x86_64-linux-gnu | FileCheck %s --check-prefix=CHECK --check-prefix=CXX11
// RUN: %clang_cc1 -std=c++1z %s -emit-llvm -o - -triple x86_64-linux-gnu | FileCheck %s --check-prefix=CHECK --check-prefix=CXX17

struct A {
  static const int Foo = 123;
};
// CHECK: @_ZN1A3FooE = constant i32 123, align 4
const int *p = &A::Foo; // emit available_externally
const int A::Foo;       // convert to full definition

struct Bar {
  int b;
};

struct MutableBar {
  mutable int b;
};

struct Foo {
  // CXX11: @_ZN3Foo21ConstexprStaticMemberE = available_externally constant i32 42,
  // CXX17: @_ZN3Foo21ConstexprStaticMemberE = linkonce_odr constant i32 42,
  static constexpr int ConstexprStaticMember = 42;
  // CHECK: @_ZN3Foo17ConstStaticMemberE = available_externally constant i32 43,
  static const int ConstStaticMember = 43;

  // CXX11: @_ZN3Foo23ConstStaticStructMemberE = available_externally constant %struct.Bar { i32 44 },
  // CXX17: @_ZN3Foo23ConstStaticStructMemberE = linkonce_odr constant %struct.Bar { i32 44 },
  static constexpr Bar ConstStaticStructMember = {44};

  // CXX11: @_ZN3Foo34ConstexprStaticMutableStructMemberE = external global %struct.MutableBar,
  // CXX17: @_ZN3Foo34ConstexprStaticMutableStructMemberE = linkonce_odr global %struct.MutableBar { i32 45 },
  static constexpr MutableBar ConstexprStaticMutableStructMember = {45};
};
// CHECK: @_ZL15ConstStaticexpr = internal constant i32 46,
static constexpr int ConstStaticexpr = 46;
// CHECK: @_ZL9ConstExpr = internal constant i32 46, align 4
static const int ConstExpr = 46;

// CHECK: @_ZL21ConstexprStaticStruct = internal constant %struct.Bar { i32 47 },
static constexpr Bar ConstexprStaticStruct = {47};

// CHECK: @_ZL28ConstexprStaticMutableStruct = internal global %struct.MutableBar { i32 48 },
static constexpr MutableBar ConstexprStaticMutableStruct = {48};

void use(const int &);
void foo() {
  use(Foo::ConstexprStaticMember);
  use(Foo::ConstStaticMember);
  use(Foo::ConstStaticStructMember.b);
  use(Foo::ConstexprStaticMutableStructMember.b);
  use(ConstStaticexpr);
  use(ConstExpr);
  use(ConstexprStaticStruct.b);
  use(ConstexprStaticMutableStruct.b);
}
