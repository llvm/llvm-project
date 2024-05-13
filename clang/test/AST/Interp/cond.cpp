// RUN: %clang_cc1 -std=c++17 -fsyntax-only -fexperimental-new-constant-interpreter %s -verify
// RUN: %clang_cc1 -std=c++17 -fsyntax-only %s -verify
// expected-no-diagnostics

constexpr int cond_then_else(int a, int b) {
  if (a < b) {
    return b - a;
  } else {
    return a - b;
  }
}

constexpr int dontCallMe(unsigned m) {
  if (m == 0) return 0;
  return dontCallMe(m - 2);
}

// Can't call this because it will run into infinite recursion.
constexpr int assertNotReached() {
  return dontCallMe(3);
}

static_assert(true || true, "");
static_assert(true || false, "");
static_assert(false || true, "");
static_assert(!(false || false), "");

static_assert(true || assertNotReached(), "");
static_assert(true || true || true || false, "");

static_assert(true && true, "");
static_assert(!(true && false), "");
static_assert(!(false && true), "");
static_assert(!(false && false), "");

static_assert(!(false && assertNotReached()), "");
static_assert(!(true && true && true && false), "");
