// RUN: %clang_cc1 -fexperimental-new-constant-interpreter -std=c++20 -verify %s
// RUN: %clang_cc1 -std=c++20 -verify=ref %s


// expected-no-diagnostics
// ref-no-diagnostics
constexpr int getMinus5() {
  int a = 10;
  a = -5;
  int *p = &a;
  return *p;
}
//static_assert(getMinus5() == -5, "") TODO

constexpr int assign() {
  int m = 10;
  int k = 12;

  m = (k = 20);

  return m;
}
//static_assert(assign() == 20, "");  TODO


constexpr int pointerAssign() {
  int m = 10;
  int *p = &m;

  *p = 12; // modifies m

  return m;
}
//static_assert(pointerAssign() == 12, "");  TODO

constexpr int pointerDeref() {
  int m = 12;
  int *p = &m;

  return *p;
}
//static_assert(pointerDeref() == 12, ""); TODO

constexpr int pointerAssign2() {
  int m = 10;
  int *p = &m;
  int **pp = &p;

  **pp = 12;

  int v = **pp;

  return v;
}
//static_assert(pointerAssign2() == 12, ""); TODO
