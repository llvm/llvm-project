// RUN: %clang_cc1 -std=c++20 -verify %s

// This test makes sure that a single element array doesn't produce
// spurious errors during constexpr evaluation.

// expected-no-diagnostics
struct Sub { int x; };

struct S {
  constexpr S() { Arr[0] = Sub{}; }
  Sub Arr[1];
};

constexpr bool test() {
  S s;
  return true;
}

static_assert(test());
