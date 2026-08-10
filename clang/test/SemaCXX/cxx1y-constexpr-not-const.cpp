// RUN: %clang_cc1 -std=c++11 %s -verify
// RUN: %clang_cc1 -std=c++1y %s -verify -DCXX1Y

struct X {
  constexpr int f(); // @5
  int f();           // @6
};

#ifdef CXX1Y
// FIXME: Detect this situation and provide a better recovery.

// expected-error@6 {{class member cannot be redeclared}}
// expected-note@5 {{previous}}
#else
// expected-warning@5 {{'constexpr' non-static member function will not be implicitly 'const' in C++14; add 'const' to avoid a change in behavior}}
#endif
