// RUN: %clang_cc1 -fsyntax-only -pedantic -Wno-strict-prototypes -verify %s

/* This test checks the introduction of struct and union types based
   on a type specifier of the form "struct-or-union identifier" when they
   type has not yet been declared. See C99 6.7.2.3p8. */

typedef struct S1 {
  union {
    struct S2 *x;
    struct S3 *y;
  } u1;
} S1;

int test_struct_scope(S1 *s1, struct S2 *s2, struct S3 *s3) {
  if (s1->u1.x == s2) return 1;
  if (s1->u1.y == s3) return 1;
  return 0;
}

int test_struct_scope_2(S1 *s1) {
  struct S2 { int x; } *s2 = 0;
  if (s1->u1.x == s2) return 1; /* expected-warning {{comparison of distinct pointer types ('struct S2 *' and 'struct S2 *')}} */
  return 0;
}

// FIXME: We do not properly implement C99 6.2.1p4, which says that
// the type "struct S4" declared in the function parameter list has
// block scope within the function definition. The problem, in this
// case, is that the code is ill-formed but we warn about the two S4's
// being incompatible (we think they are two different types).
int test_struct_scope_3(struct S4 * s4) { // expected-warning{{declaration of 'struct S4' will not be visible outside of this function}}
  struct S4 { int y; } *s4_2 = 0;
  /*  if (s4 == s4_2) return 1; */
  return 0;
}

void f(struct S5 { int y; } s5); // expected-warning{{declaration of 'struct S5' will not be visible outside of this function}}

// PR clang/3312
struct S6 {
        enum { BAR } e;
};

void test_S6(void) {
        struct S6 a;
        a.e = BAR;
}

// <rdar://problem/6487669>
typedef struct z_foo_s {
  struct bar_baz *baz;
} z_foo;
typedef z_foo *z_foop;
struct bar_baz {
  enum {
    SQUAT, FLAG, DICT4, DICT3, DICT2, DICT1, DICT0, HOP, CHECK4, CHECK3, CHECK2, CHECK1, DONE, BAD
  } mode;
  int             nowrap;
};
void
wizbiz_quxPoof(z)
  z_foop       z;
{
  z->baz->mode = z->baz->nowrap ? HOP : SQUAT;
}
