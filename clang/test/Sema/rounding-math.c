// RUN: %clang_cc1 -triple x86_64-linux -verify=norounding %s
// RUN: %clang_cc1 -triple x86_64-linux -std=c17 -verify=rounding-std %s -frounding-math
// RUN: %clang_cc1 -triple x86_64-linux -std=gnu17 -verify=rounding-gnu %s -frounding-math

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
  // Under -frounding-math, this is a VLA.
  // FIXME: Due to PR44406, in GNU mode we constant-fold the initializer resulting in a non-VLA.
  typedef int vla[(int)(-3 * (1.0 / 3.0))]; // norounding-error {{negative size}} rounding-gnu-error {{negative size}}
  struct X { vla v; }; // rounding-std-error {{fields must have a constant size}}
}
