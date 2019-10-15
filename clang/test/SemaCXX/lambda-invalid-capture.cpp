// RUN: %clang_cc1 -fsyntax-only -verify %s
// Don't crash.

struct g {
  j; // expected-error {{C++ requires a type specifier for all declarations}}
};

void captures_invalid_type() {
  g child;
  auto q = [child]{};
  const int n = sizeof(q);
}

void captures_invalid_array_type() {
  g child[100];
  auto q = [child]{};
  const int n = sizeof(q);
}

int pr43080(int i) { // expected-note {{declared here}}
  return [] { // expected-note {{begins here}}
    return sizeof i <
      i; // expected-error {{variable 'i' cannot be implicitly captured in a lambda with no capture-default specified}}
  }();
}
