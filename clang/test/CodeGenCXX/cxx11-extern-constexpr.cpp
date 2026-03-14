// RUN: %clang_cc1 -std=c++11 %s -emit-llvm -o - -triple x86_64-linux-gnu | FileCheck %s --check-prefixes=X86,CXX11X86
// RUN: %clang_cc1 -std=c++1z %s -emit-llvm -o - -triple x86_64-linux-gnu | FileCheck %s --check-prefixes=X86,CXX17X86
// RUN: %clang_cc1 -std=c++11 %s -emit-llvm -o - -triple amdgcn-amd-amdhsa | FileCheck %s --check-prefixes=AMD,CXX11AMD
// RUN: %clang_cc1 -std=c++1z %s -emit-llvm -o - -triple amdgcn-amd-amdhsa | FileCheck %s --check-prefixes=AMD,CXX17AMD

struct A {
  static const int Foo = 123;
};
// X86: @_ZN1A3FooE ={{.*}} constant i32 123, align 4
// AMD: @_ZN1A3FooE ={{.*}} addrspace(1) constant i32 123, align 4
const int *p = &A::Foo; // emit available_externally
const int A::Foo;       // convert to full definition

struct PODWithInit {
  int g = 42;
  char h = 43;
};
struct CreatePOD {
  // Deferred initialization of the structure here requires changing
  // the type of the global variable: the initializer list does not include
  // the tail padding.
  // CXX11X86: @_ZN9CreatePOD3podE = available_externally constant { i32, i8 } { i32 42, i8 43 },
  // CXX11AMD: @_ZN9CreatePOD3podE = available_externally addrspace(1) constant { i32, i8 } { i32 42, i8 43 },
  static constexpr PODWithInit pod{};
};
const int *p_pod = &CreatePOD::pod.g;

struct Bar {
  int b;
};

struct MutableBar {
  mutable int b;
};

struct Foo {
  // CXX11X86: @_ZN3Foo21ConstexprStaticMemberE = available_externally constant i32 42,
  // CXX17X86: @_ZN3Foo21ConstexprStaticMemberE = linkonce_odr constant i32 42,
  // CXX11AMD: @_ZN3Foo21ConstexprStaticMemberE = available_externally addrspace(4) constant i32 42,
  // CXX17AMD: @_ZN3Foo21ConstexprStaticMemberE = linkonce_odr addrspace(4) constant i32 42, comdat, align 4
  static constexpr int ConstexprStaticMember = 42;
  // X86: @_ZN3Foo17ConstStaticMemberE = available_externally constant i32 43,
  // AMD: @_ZN3Foo17ConstStaticMemberE = available_externally addrspace(4) constant i32 43,
  static const int ConstStaticMember = 43;

  // CXX11X86: @_ZN3Foo23ConstStaticStructMemberE = available_externally constant %struct.Bar { i32 44 },
  // CXX17X86: @_ZN3Foo23ConstStaticStructMemberE = linkonce_odr constant %struct.Bar { i32 44 },
  // CXX11AMD: @_ZN3Foo23ConstStaticStructMemberE = available_externally addrspace(1) constant %struct.Bar { i32 44 },
  // CXX17AMD: @_ZN3Foo23ConstStaticStructMemberE = linkonce_odr addrspace(1) constant %struct.Bar { i32 44 },
  static constexpr Bar ConstStaticStructMember = {44};

  // CXX11X86: @_ZN3Foo34ConstexprStaticMutableStructMemberE = external global %struct.MutableBar,
  // CXX17X86: @_ZN3Foo34ConstexprStaticMutableStructMemberE = linkonce_odr global %struct.MutableBar { i32 45 },
  // CXX11AMD: @_ZN3Foo34ConstexprStaticMutableStructMemberE = external addrspace(1) global %struct.MutableBar,
  // CXX17AMD: @_ZN3Foo34ConstexprStaticMutableStructMemberE = linkonce_odr addrspace(1) global %struct.MutableBar { i32 45 },
  static constexpr MutableBar ConstexprStaticMutableStructMember = {45};
};
// X86: @_ZL15ConstStaticexpr = internal constant i32 46,
// AMD: @_ZL15ConstStaticexpr = internal addrspace(4) constant i32 46,
static constexpr int ConstStaticexpr = 46;
// X86: @_ZL9ConstExpr = internal constant i32 46, align 4
// AMD: @_ZL9ConstExpr = internal addrspace(4) constant i32 46, align 4
static const int ConstExpr = 46;

// X86: @_ZL21ConstexprStaticStruct = internal constant %struct.Bar { i32 47 },
// AMD: @_ZL21ConstexprStaticStruct = internal addrspace(1) constant %struct.Bar { i32 47 },
static constexpr Bar ConstexprStaticStruct = {47};

// X86: @_ZL28ConstexprStaticMutableStruct = internal global %struct.MutableBar { i32 48 },
// AMD: @_ZL28ConstexprStaticMutableStruct = internal addrspace(1) global %struct.MutableBar { i32 48 },
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
