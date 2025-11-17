// RUN: %check_clang_tidy %s misc-shadowed-namespace-function %t

int f1();
namespace foo {
  char f0();
  unsigned f1();
}
short f0() { return {}; }
int f1() { return {}; }
