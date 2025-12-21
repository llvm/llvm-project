// RUN: %clang_cc1 -emit-pch -std=c++23 -o %t %s
// RUN: %clang_cc1 -include-pch %t -verify -fsyntax-only -DTEST -std=c++23 %s

// Test that dependence of 'this' and DREs due to by-value capture by a
// lambda with an explicit object parameter is serialised/deserialised
// properly.

#ifndef HEADER
#define HEADER
struct S {
  int x;
  auto f() {
    return [*this] (this auto&&) {
      int y;
      x = 42;

      const auto l = [y] (this auto&&) { y = 42; };
      l();
    };
  }
};
#endif

// expected-error@* {{read-only variable is not assignable}}
// expected-error@* {{cannot assign to a variable captured by copy in a non-mutable lambda}}
// expected-note@* 2 {{in instantiation of}}

#ifdef TEST
void f() {
  const auto l = S{}.f();
  l(); // expected-note {{in instantiation of}}
}
#endif


