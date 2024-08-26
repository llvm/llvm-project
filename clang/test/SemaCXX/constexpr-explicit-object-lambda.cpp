// RUN: %clang_cc1 -std=c++23 -verify %s
// expected-no-diagnostics

struct S {
  int i = 42;
  constexpr auto f1() {
    return [this](this auto) {
      return this->i;
    }();
  };

  constexpr auto f2() {
    return [this](this auto&&) {
      return this->i;
    }();
  };

  constexpr auto f3() {
    return [i = this->i](this auto) {
      return i;
    }();
  };

  constexpr auto f4() {
    return [i = this->i](this auto&&) {
      return i;
    }();
  };
};

static_assert(S().f1() == 42);
static_assert(S().f2() == 42);
static_assert(S().f3() == 42);
static_assert(S().f4() == 42);
