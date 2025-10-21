// RUN: %clang_cc1 %s -std=c++20 -verify=both,expected -fexperimental-new-constant-interpreter

// both-no-diagnostics


constexpr int F[100] = {1,2};
static_assert(F[98] == 0);
static_assert(F[99] == 0);
static_assert(F[0] == 1);
static_assert(F[1] == 2);
static_assert(F[2] == 0);

constexpr _Complex double Doubles[4] = {{1.0, 2.0}};
static_assert(__real(Doubles[0]) == 1.0, "");
static_assert(__imag(Doubles[0]) == 2.0, "");

static_assert(__real(Doubles[1]) == 0.0, "");
static_assert(__imag(Doubles[1]) == 0.0, "");

static_assert(__real(Doubles[2]) == 0.0, "");
static_assert(__imag(Doubles[2]) == 0.0, "");
static_assert(__real(Doubles[3]) == 0.0, "");
static_assert(__imag(Doubles[3]) == 0.0, "");

static_assert(__real(Doubles[0]) == 1.0, "");
static_assert(__imag(Doubles[0]) == 2.0, "");

struct S {
  int x = 20;
};
constexpr S s[20] = {};
static_assert(s[0].x == 20);
static_assert(s[1].x == 20);
static_assert(s[2].x == 20);
static_assert(s[3].x == 20);
static_assert(s[4].x == 20);

constexpr int test() {
  int a[4] = {};
  int r = a[2];
  return r;
}
static_assert(test() == 0);

constexpr int test2() {
  char buff[2] = {};
  buff[0] = 'B';
  return buff[1] == '\0' && buff[0] == 'B';
}
static_assert(test2());
