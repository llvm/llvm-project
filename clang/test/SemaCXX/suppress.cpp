// RUN: %clang_cc1 -std=c++11 -fsyntax-only %s -verify

[[gsl::suppress("globally")]];

namespace N {
  [[gsl::suppress("in-a-namespace")]];
}

[[gsl::suppress("readability-identifier-naming")]]
void f_() {
  int *p;
  [[gsl::suppress("type", "bounds")]] {
    p = reinterpret_cast<int *>(7);
  }

  [[gsl::suppress]] int x; // expected-error {{'suppress' attribute takes at least 1 argument}}
  [[gsl::suppress()]] int y; // expected-error {{'suppress' attribute takes at least 1 argument}}
  int [[gsl::suppress("r")]] z; // expected-error {{'suppress' attribute cannot be applied to types}}
  [[gsl::suppress(f_)]] float f; // expected-error {{expected string literal as argument of 'suppress' attribut}}
}

union [[gsl::suppress("type.1")]] U {
  int i;
  float f;
};
