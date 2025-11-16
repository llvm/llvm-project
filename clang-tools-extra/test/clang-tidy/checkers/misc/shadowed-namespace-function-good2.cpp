// RUN: %check_clang_tidy %s misc-shadowed-namespace-function %t

void f1();
namespace foo {
  void f0();
  void f1();
}
void foo::f0() {}
void foo::f1() {}
void f0() {}
void f1() {}
