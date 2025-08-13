// RUN: %clang_cc1 -fsyntax-only -verify %s

void foo() {
  struct Local {
    template <typename T> // expected-error {{member templates are not allowed inside local classes}}
    void bar();
  };
}

