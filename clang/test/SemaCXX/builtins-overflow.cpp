// RUN: %clang_cc1 -fsyntax-only -std=c++17 -verify %s
// expected-no-diagnostics

int a() {
  const int x = 3;
  static int z;
  constexpr int *y = &z;
  return []() { return __builtin_sub_overflow((int)x, (int)x, (int *)y); }();
}
int a2() {
  const int x = 3;
  static int z;
  constexpr int *y = &z;
  return []() { return __builtin_sub_overflow(x, x, y); }();
}
