// RUN: %clang_cc1 -std=c++20 -fsyntax-only -verify %s
// RUN: %clang_cc1 -std=c++23 -fsyntax-only -verify %s
// expected-no-diagnostics

constexpr bool integers(unsigned n) {
  int *p = new int[n](5);
  bool result = p[0] == 5;
  for (unsigned i = 1; i != n; ++i)
    result &= p[i] == 0;
  delete[] p;
  return result;
}
static_assert(integers(1));
static_assert(integers(3));
static_assert(integers(5));

struct Element {
  int value;
  constexpr Element(int value = 7) : value(value) {}
};
constexpr bool objects(unsigned n) {
  Element *p = new Element[n](Element(1), Element(2));
  bool result = p[0].value == 1 && p[1].value == 2;
  for (unsigned i = 2; i != n; ++i)
    result &= p[i].value == 7;
  delete[] p;
  return result;
}
static_assert(objects(2));
static_assert(objects(4));

constexpr bool string(unsigned n) {
  char *p = new char[n]{"abc"};
  bool result = p[0] == 'a' && p[3] == 0 && p[n - 1] == 0;
  delete[] p;
  return result;
}
static_assert(string(4));
static_assert(string(8));
