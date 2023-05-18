// RUN: %clang_cc1 -Wno-uninitialized -std=c++1y %s -verify

// expected-no-diagnostics

struct S { int a; const char *b; int c; int d = b[a]; };
constexpr S ss = { 1, "asdf" };

static_assert(ss.a == 1, "");
static_assert(ss.b[2] == 'd', "");
static_assert(ss.c == 0, "");
static_assert(ss.d == 's', "");

struct X { int i, j, k = 42; };
constexpr X a[] = { 1, 2, 3, 4, 5, 6 };
constexpr X b[2] = { { 1, 2, 3 }, { 4, 5, 6 } };

constexpr bool operator==(X a, X b) {
  return a.i == b.i && a.j == b.j && a.k == b.k;
}

static_assert(sizeof(a) == sizeof(b), "");
static_assert(a[0] == b[0], "");
static_assert(a[1] == b[1], "");
