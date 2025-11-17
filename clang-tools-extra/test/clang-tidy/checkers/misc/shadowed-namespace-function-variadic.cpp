// RUN: %check_clang_tidy %s misc-shadowed-namespace-function %t

void f1(...);
namespace foo {
  void f0(...);
  void f1(...);
}

// FIXME: warning in these two cases??
// FIXME: fixit for f0??
void f0(...) {}
void f1(...) {}
