// RUN: %clang_cc1 -std=c++23 -fsyntax-only -verify %s

template<class>
struct D;

template<class T>
void foo(D<T>);

template<class T>
struct D {
  friend void ::foo(D) {} // expected-error {{friend function definition cannot be qualified with '::'}}
};

int main() {
  foo(D<int>{});
}

