// RUN: %clang_cc1 -fsyntax-only -verify %s
// expected-no-diagnostics

// Make sure that friend declarations don't introduce ambiguous
// declarations.

// Test case courtesy of Shantonu Sen.
// Bug 4784.

class foo;

extern "C" {
  int c_func(foo *a);
};
int cpp_func(foo *a);

class foo {
public:
  friend int c_func(foo *a);
  friend int cpp_func(foo *a);
  int caller();
private:
  int x;
};

int c_func(foo *a) {
  return a->x;
}

int cpp_func(foo *a) {
  return a->x;
}

int foo::caller() {
    c_func(this);
    cpp_func(this);
    return 0;
}
