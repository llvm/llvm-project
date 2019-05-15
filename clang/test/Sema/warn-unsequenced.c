// RUN: %clang_cc1 -fsyntax-only -verify -std=c11 -Wno-unused %s

int f(int, int);

typedef struct A {
  int x, y;
} A;

void test() {
  int a;
  int xs[10];
  a + ++a; // expected-warning {{unsequenced modification and access to 'a'}}
  a = ++a; // expected-warning {{multiple unsequenced modifications to 'a'}}
  a + a++; // expected-warning {{unsequenced modification and access to 'a'}}
  a = a++; // expected-warning {{multiple unsequenced modifications to 'a'}}
  (a++, a++); // ok
  ++a + ++a; // expected-warning {{multiple unsequenced modifications}}
  a++ + a++; // expected-warning {{multiple unsequenced modifications}}
  a = xs[++a]; // expected-warning {{multiple unsequenced modifications}}
  a = xs[a++]; // expected-warning {{multiple unsequenced modifications}}
  a = (++a, ++a); // expected-warning {{multiple unsequenced modifications}}
  a = (a++, ++a); // expected-warning {{multiple unsequenced modifications}}
  a = (a++, a++); // expected-warning {{multiple unsequenced modifications}}
  f(a, a); // ok
  f(a = 0, a); // expected-warning {{unsequenced modification and access}}
  f(a, a += 0); // expected-warning {{unsequenced modification and access}}
  f(a = 0, a = 0); // expected-warning {{multiple unsequenced modifications}}
  a = f(++a, 0); // ok
  a = f(a++, 0); // ok
  a = f(++a, a++); // expected-warning {{multiple unsequenced modifications}}

  ++a + f(++a, 0); // expected-warning {{multiple unsequenced modifications}}
  f(++a, 0) + ++a; // expected-warning {{multiple unsequenced modifications}}
  a++ + f(a++, 0); // expected-warning {{multiple unsequenced modifications}}
  f(a++, 0) + a++; // expected-warning {{multiple unsequenced modifications}}

  a = ++a; // expected-warning {{multiple unsequenced modifications}}
  a += ++a; // expected-warning {{unsequenced modification and access}}

  A agg1 = { a++, a++ }; // expected-warning {{multiple unsequenced modifications}}
  A agg2 = { a++ + a, a++ }; // expected-warning {{unsequenced modification and access}}

  (xs[2] && (a = 0)) + a; // ok
  (0 && (a = 0)) + a; // ok
  (1 && (a = 0)) + a; // expected-warning {{unsequenced modification and access}}

  (xs[3] || (a = 0)) + a; // ok
  (0 || (a = 0)) + a; // expected-warning {{unsequenced modification and access}}
  (1 || (a = 0)) + a; // ok

  (xs[4] ? a : ++a) + a; // ok
  (0 ? a : ++a) + a; // expected-warning {{unsequenced modification and access}}
  (1 ? a : ++a) + a; // ok
  (xs[5] ? ++a : ++a) + a; // FIXME: warn here

  (++a, xs[6] ? ++a : 0) + a; // expected-warning {{unsequenced modification and access}}

  // Here, the read of the fourth 'a' might happen before or after the write to
  // the second 'a'.
  a += (a++, a) + a; // expected-warning {{unsequenced modification and access}}

  int *p = xs;
  a = *(a++, p); // ok
  a = a++ && a; // ok

  A *q = &agg1;
  (q = &agg2)->y = q->x; // expected-warning {{unsequenced modification and access to 'q'}}

  // This has undefined behavior if a == 0; otherwise, the side-effect of the
  // increment is sequenced before the value computation of 'f(a, a)', which is
  // sequenced before the value computation of the '&&', which is sequenced
  // before the assignment. We treat the sequencing in '&&' as being
  // unconditional.
  a = a++ && f(a, a);

  // This has undefined behavior if a != 0. FIXME: We should diagnose this.
  (a && a++) + a;

  (xs[7] && ++a) * (!xs[7] && ++a); // ok

  xs[0] = (a = 1, a); // ok

  xs[8] ? ++a + a++ : 0; // expected-warning {{multiple unsequenced modifications}}
  xs[8] ? 0 : ++a + a++; // expected-warning {{multiple unsequenced modifications}}
  xs[8] ? ++a : a++; // ok

  xs[8] && (++a + a++); // expected-warning {{multiple unsequenced modifications}}
  xs[8] || (++a + a++); // expected-warning {{multiple unsequenced modifications}}

  (__builtin_classify_type(++a) ? 1 : 0) + ++a; // ok
  (__builtin_constant_p(++a) ? 1 : 0) + ++a; // ok
  (__builtin_expect(++a, 0) ? 1 : 0) + ++a; // expected-warning {{multiple unsequenced modifications}}
  _Generic(++a, default: 0) + ++a; // ok
  sizeof(++a) + ++a; // ok
  _Alignof(++a) + ++a; // expected-warning {{extension}}

  __builtin_constant_p(f(++a, 0)) ? f(f(++a, 0), f(++a, 0)) : 0;

  if (0) ++a + ++a; // ok, unreachable
}

void g(const char *p, int n) {
  // This resembles code produced by some macros in glibc's <string.h>.
  __builtin_constant_p(p) && __builtin_constant_p(++n) && (++n + ++n);
}
