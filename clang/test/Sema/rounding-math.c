// RUN: %clang_cc1 -triple x86_64-linux -std=c17 -verify=expected,norounding %s
// RUN: %clang_cc1 -triple x86_64-linux -std=gnu17 -verify=expected,norounding %s
// RUN: %clang_cc1 -triple x86_64-linux -std=c17 -verify=expected,rounding %s -frounding-math
// RUN: %clang_cc1 -triple x86_64-linux -std=gnu17 -verify=expected,rounding %s -frounding-math

#define fold(x) (__builtin_constant_p(x) ? (x) : (x))

double a = 1.0 / 3.0;

#define f(n) ((n) * (1.0 / 3.0))
_Static_assert(fold((int)f(3)) == 1, "");

typedef int T[fold((int)f(3))];
typedef int T[1];

enum Enum { enum_a = (int)f(3) };

struct Bitfield {
  unsigned int n : 1;
  unsigned int m : fold((int)f(3));
};

void bitfield(struct Bitfield *b) {
  b->n = (int)(6 * (1.0 / 3.0)); // norounding-warning {{changes value from 2 to 0}}
}

void vlas() {
  // This is always a VLA due to its syntactic form.
  typedef int vla1[(int)(-3 * (1.0 / 3.0))];
  struct X1 { vla1 v; }; // expected-error {{fields must have a constant size}}

  // This is always folded to a constant.
  typedef int vla2[fold((int)(-3 * (1.0 / 3.0)))]; // expected-error {{negative size}}
  struct X2 { vla2 v; };
}
