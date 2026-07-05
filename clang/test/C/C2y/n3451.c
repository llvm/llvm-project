// RUN: %clang_cc1 -verify -std=c2y -Wall -pedantic -Wno-gnu-folding-constant %s
// RUN: %clang_cc1 -verify -std=c23 -Wall -pedantic -Wno-gnu-folding-constant %s

/* WG14 N3451: Yes
 * Initialization of anonymous structures and unions
 *
 * This paper allows initialization of anonymous structure and union members
 * within the containing object.
 */
// expected-no-diagnostics

constexpr struct {
  int a : 10;
  int : 12;
  long b;
} s = { 1, 2 };
static_assert(s.a == 1 && s.b == 2);

constexpr union {
  int : 16;
  char c;
} u = {3};
static_assert(u.c == 3);

constexpr struct {
  union {
    float a;
    int b;
    void *p;
  };
  char c;
} t = {{.b = 1}, 2};
static_assert(t.b == 1 && t.c == 2);

constexpr struct {
  union {
    float a;
    int b;
    void *p;
  };
  char c;
} v = {.b = 1, 2};
static_assert(v.b == 1 && v.c == 2);
