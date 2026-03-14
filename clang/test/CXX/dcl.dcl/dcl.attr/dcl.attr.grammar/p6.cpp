// RUN: %clang_cc1 -std=c++11 -verify %s

namespace std_example {

int p[10];
void f() {
  int x = 42, y[5];
  // FIXME: Produce a better diagnostic for this case.
  int(p[[x] { return x; }()]); // expected-error {{expected ']'}} \
  // expected-warning {{unknown attribute 'x' ignored}}
  y[[] { return 2; }()] = 2; // expected-error {{consecutive left square brackets}}
}

}
