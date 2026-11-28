// RUN: %clang_cc1 -fsyntax-only -verify -std=c++98 %s
// RUN: %clang_cc1 -fsyntax-only -verify -std=c++11 %s
// RUN: %clang_cc1 -fsyntax-only -verify -std=c++17 %s
// RUN: %clang_cc1 -fsyntax-only -verify -std=c++20 %s

// Used to hit "Cannot get layout of invalid decl!" because the invalid
// specialization started a second definition. See GH207953.

template <typename T>
struct X {
  struct Y {
    Y() : v(0) {}
    int v;
    int getValue();
  } y; // expected-note {{implicit instantiation first required here}}
};

template <typename T>
int X<T>::Y::getValue() {
  return ++v;
}

template <> struct X<int>::Y { int getValue() { return 55; } };
// expected-error@-1 {{explicit specialization of 'Y' after instantiation}}

extern template class X<int>::Y;

int main() {
  X<int> x;
  return x.y.getValue();
}
