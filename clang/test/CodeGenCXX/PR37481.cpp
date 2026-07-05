// RUN: %clang_cc1 -o /dev/null -emit-llvm -std=c++17 -triple x86_64-pc-windows-msvc %s

struct Foo {
  virtual void f();
  virtual void g();
};

void Foo::f() {}
void Foo::g() {}

template <void (Foo::*)()>
void h() {}

void x() {
  h<&Foo::f>();
  h<&Foo::g>();
}
