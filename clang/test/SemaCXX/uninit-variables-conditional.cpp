// RUN: %clang_cc1 -Wconditional-uninitialized -fsyntax-only %s -verify

class Foo {
public:
  Foo();
  ~Foo();
  operator bool();
};

int bar();
int baz();
int init(double *);

// This case flags a false positive under -Wconditional-uninitialized because
// the destructor in Foo fouls about the minor bit of path-sensitivity in
// -Wuninitialized.
double test() {
  double x; // expected-note{{initialize the variable 'x' to silence this warning}}
  if (bar() || baz() || Foo() || init(&x))
    return 1.0;

  return x; // expected-warning {{variable 'x' may be uninitialized when used here}}
}
