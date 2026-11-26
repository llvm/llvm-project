// RUN: %clang_cc1 -std=c++11 -fsyntax-only -verify %s

// Diagnose modification of a parameter declared const in an out-of-line member
// definition, even if the earlier declaration omits const.
template <typename T>
struct X {
  int f(X &&, int);

  int value;
};

template <typename T>
int X<T>::f(X &&u, const int x) { // expected-note {{variable 'x' declared const here}}
  x += u.value; // expected-error {{cannot assign to variable 'x' with const-qualified type 'const int'}}
  return x;
}

void test() {
  X<int>{}.f(X<int>{1}, 0);
}
