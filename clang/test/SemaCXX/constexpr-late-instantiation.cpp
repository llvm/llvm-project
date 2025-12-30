// RUN: %clang_cc1 %s -fsyntax-only -verify
// RUN: %clang_cc1 %s -fexperimental-new-constant-interpreter -fsyntax-only -verify

template <typename T>
constexpr T foo(T a);   // expected-note {{declared here}}

int main() {
  int k = foo<int>(5);  // Ok
  constexpr int j =     // expected-error {{constexpr variable 'j' must be initialized by a constant expression}}
          foo<int>(5);  // expected-note {{undefined function 'foo<int>' cannot be used in a constant expression}}
}

template <typename T>
constexpr T foo(T a) {
  return a;
}
