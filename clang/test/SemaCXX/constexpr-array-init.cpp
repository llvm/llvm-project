// RUN: %clang_cc1 -std=c++20 -verify %s

/// This test case used to crash in constant evaluation
/// because of the two-dimensional array with an array
/// filler expression.

/// expected-no-diagnostics
struct Foo {
  int a;
  constexpr Foo()
      : a(get_int()) {
  }

  constexpr int get_int() const {
    return 5;
  }
};

static constexpr Foo bar[2][1] = {
    {{}},
};
static_assert(bar[0][0].a == 5);
static_assert(bar[1][0].a == 5);

