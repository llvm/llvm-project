// RUN: %check_clang_tidy %s misc-shadowed-namespace-function %t

namespace foo {
  int main();
}
int main() {}
