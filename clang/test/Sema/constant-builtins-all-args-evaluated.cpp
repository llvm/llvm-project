// RUN: %clang_cc1 -fsyntax-only -verify %s
// expected-no-diagnostics

constexpr int increment(int& x) {
  x++;
  return x;
}

constexpr int test_clzg_0() {
  int x = 0;
  (void)__builtin_clzg(0U, increment(x));
  return x;
}

static_assert(test_clzg_0() == 1);

constexpr int test_clzg_1() {
  int x = 0;
  (void)__builtin_clzg(1U, increment(x));
  return x;
}

static_assert(test_clzg_1() == 1);

constexpr int test_ctzg_0() {
  int x = 0;
  (void)__builtin_ctzg(0U, increment(x));
  return x;
}

static_assert(test_ctzg_0() == 1);

constexpr int test_ctzg_1() {
  int x = 0;
  (void)__builtin_ctzg(1U, increment(x));
  return x;
}

static_assert(test_ctzg_1() == 1);
