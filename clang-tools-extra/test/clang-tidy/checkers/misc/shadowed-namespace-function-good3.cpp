// RUN: %check_clang_tidy %s misc-shadowed-namespace-function %t

void f1(int);
namespace foo {
  void f0(short);
  void f1(unsigned);
}
void f0(char) {}
void f1(int) {}
