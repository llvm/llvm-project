// RUN: %check_clang_tidy %s misc-shadowed-namespace-function %t

static void f1();
namespace foo {
  void f0();
  void f1();
}
static void f0() {}
static void f1() {}
