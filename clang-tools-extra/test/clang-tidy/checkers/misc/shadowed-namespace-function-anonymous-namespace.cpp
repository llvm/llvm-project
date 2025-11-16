// RUN: %check_clang_tidy %s misc-shadowed-namespace-function %t

namespace { void f1(); }
namespace foo {
  void f0();
  void f1();
}
namespace {
void f0() {}
void f1() {}
}
