// RUN: %clang_cc1 -std=c++11 -verify %s

typedef int A alignas(4); // expected-error {{'alignas' attribute only applies to variables, data members and tag types}}
template<int N> void f() {
  typedef int B alignas(N); // expected-error {{'alignas' attribute only applies to variables, data members and tag types}}
}
