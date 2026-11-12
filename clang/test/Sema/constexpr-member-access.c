// RUN: %clang_cc1 -std=c23 -verify -triple x86_64 -pedantic -Wno-conversion -Wno-constant-conversion -Wno-div-by-zero %s
// RUN: %clang_cc1 -std=c23 -verify -triple x86_64 -pedantic -Wno-conversion -Wno-constant-conversion -Wno-div-by-zero -fexperimental-new-constant-interpreter %s

#define comptime_if(predicate, on_true, on_false)                              \
  _Generic((char (*)[1 + !!(predicate)]){0},                                   \
      char (*)[2]: (on_true),                                                  \
      char (*)[1]: (on_false))

enum E { E_A = 0, E_B = 1 };
union U {
  int a;
  char b;
};
struct S1 {
  int a;
};
struct S2 {
  int a;
  int b;
  _Bool c;
  char d;
  enum E e;
  struct S1 f;
  double g;
  int h;
};

constexpr struct S2 V1 = {0, 1, 1, 'c', E_B, {3}, 1.0, -1};
constexpr union U V2 = {5};
constexpr int V3 = V1.f.a;
constexpr int V4 = ((struct S2){0, 4, 0, 0, E_A, {6}, 0.0, 0}).f.a;

void gh178349() {
  int a[V1.b] = {};
  int b[V1.g] = {}; // expected-error {{size of array has non-integer type 'double'}}
  int c[V1.h] = {}; // expected-error {{'c' declared as an array with a negative size}}

  const struct S2 *P1 = &V1;
  _Static_assert(P1->b, ""); // expected-error {{static assertion expression is not an integral constant expression}}
  _Static_assert(V1.b, "");

  _Static_assert(comptime_if(V1.a, 1, 0) == 0, "");
  _Static_assert(comptime_if(V1.a, 0, 1) == 1, "");
  _Static_assert(comptime_if(V2.a, 1, 0) == 1, "");

  _Static_assert(V1.c, "");
  _Static_assert(V1.d == 'c', "");
  _Static_assert(V1.e == E_B, "");
  _Static_assert(V1.f.a == 3, "");

  _Static_assert(V2.a == 5, "");
  _Static_assert(V3 == 3, "");
  _Static_assert(V4 == 6, "");
}
