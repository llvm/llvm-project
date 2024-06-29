// RUN: %clang_cc1 -std=c++11 -fsyntax-only %s -verify

[[gsl::suppress("globally")]];

namespace N {
[[gsl::suppress("in-a-namespace")]];
}

[[gsl::suppress("readability-identifier-naming")]] void f_() {
  int *p;
  [[gsl::suppress("type", "bounds")]] {
    p = reinterpret_cast<int *>(7);
  }

  [[gsl::suppress]] int x;       // expected-error {{'suppress' attribute takes at least 1 argument}}
  [[gsl::suppress()]] int y;     // expected-error {{'suppress' attribute takes at least 1 argument}}
  int [[gsl::suppress("r")]] z;  // expected-error {{'suppress' attribute cannot be applied to types}}
  [[gsl::suppress(f_)]] float f; // expected-error {{expected string literal as argument of 'suppress' attribute}}
}

union [[gsl::suppress("type.1")]] U {
  int i;
  float f;
};

// This doesn't really suppress anything but why not?
[[clang::suppress]];

namespace N {
[[clang::suppress("in-a-namespace")]];
} // namespace N

[[clang::suppress]] int global = 42;

[[clang::suppress]] void foo() {
  [[clang::suppress]] int *p;

  [[clang::suppress]] int a = 0;           // no-warning
  [[clang::suppress()]] int b = 1;         // no-warning
  [[clang::suppress("a")]] int c = a + b;  // no-warning
  [[clang::suppress("a", "b")]] b = c - a; // no-warning

  [[clang::suppress("a", "b")]] if (b == 10) a += 4; // no-warning
  [[clang::suppress]] while (true) {}                // no-warning
  [[clang::suppress]] switch (a) {                   // no-warning
  default:
    c -= 10;
  }

  int [[clang::suppress("r")]] z;
  // expected-error@-1 {{'suppress' attribute cannot be applied to types}}
  [[clang::suppress(foo)]] float f;
  // expected-error@-1 {{expected string literal as argument of 'suppress' attribute}}
}

class [[clang::suppress("type.1")]] V {
  int i;
  float f;
};

// FIXME: There's no good reason why we shouldn't support this case.
// But it doesn't look like clang generally supports such attributes yet.
class W : [[clang::suppress]] public V { // expected-error{{'suppress' attribute cannot be applied to a base specifier}}
};
