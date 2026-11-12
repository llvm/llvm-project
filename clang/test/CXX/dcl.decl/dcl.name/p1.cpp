// RUN: %clang_cc1 -fsyntax-only -verify %s
// expected-no-diagnostics

namespace pr6200 {
  struct v {};
  enum E { e };
  struct s {
    int i;
    operator struct v() { return v(); };
    operator enum E() { return e; }
  };

  void f()
  {
    // None of these is a declaration.
    (void)new struct s;
    (void)new enum E;
    (void)&s::operator struct v;
    (void)&s::operator enum E;
  }
}
