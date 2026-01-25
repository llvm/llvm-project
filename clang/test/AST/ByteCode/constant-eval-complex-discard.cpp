// RUN: %clang_cc1 -std=c++20 -fsyntax-only -verify -fexperimental-new-constant-interpreter %s
// RUN: %clang_cc1 -std=c++20 -fsyntax-only -verify %s

// expected-no-diagnostics   

void test_no_crash() {
  _Complex int x = 1i;
  (void)(x == 1i);
}

constexpr int test_side_effect() {
  int k = 0;
  (void)(1i == (++k, 1i));
  return k;
}
static_assert(test_side_effect() == 1);