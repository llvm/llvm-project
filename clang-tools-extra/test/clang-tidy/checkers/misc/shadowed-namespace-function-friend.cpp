// RUN: %check_clang_tidy %s misc-shadowed-namespace-function %t


namespace foo { struct A; }
void f1(foo::A);

namespace foo {
  struct A{
    friend void f0(A);
    friend void f1(A);
  };
}

// FIXME: provide warning without fixit in these two cases
void f0(foo::A) {}
void f1(foo::A) {}
