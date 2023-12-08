// RUN: %clang_cc1 -fblocks -fsyntax-only -Wunused-but-set-variable -verify %s

struct S {
  int i;
};

int f0(void) {
  int y; // expected-warning{{variable 'y' set but not used}}
  y = 0;

  int z __attribute__((unused));
  z = 0;

  struct S s; // expected-warning{{variable 's' set but not used}}
  struct S t;
  s = t;

  // Don't warn for an extern variable.
  extern int w;
  w = 0;

  // Following gcc, this should not warn.
  int a;
  w = (a = 0);

  int j = 0; // expected-warning{{variable 'j' set but not used}}
  for (int i = 0; i < 1000; i++)
    j += 1;

  // Following gcc, warn for a volatile variable.
  volatile int b; // expected-warning{{variable 'b' set but not used}}
  b = 0;

  // volatile variable k is used, no warning.
  volatile int k = 0;
  for (int i = 0; i < 1000; i++)
    k += 1;

  // typedef of volatile type, no warning.
  typedef volatile int volint;
  volint l = 0;
  l += 1;

  int x;
  x = 0;
  return x;
}

void f1(void) {
  (void)^() {
    int y; // expected-warning{{variable 'y' set but not used}}
    y = 0;

    int x;
    x = 0;
    return x;
  };
}

void f2 (void) {
  // Don't warn, even if it's only used in a non-ODR context.
  int x;
  x = 0;
  (void) sizeof(x);
}

void for_cleanup(int *x) {
  *x = 0;
}

void f3(void) {
  // Don't warn if the __cleanup__ attribute is used.
  __attribute__((__cleanup__(for_cleanup))) int x;
  x = 5;
}

void f4(void) {
  int x1 = 0; // expected-warning{{variable 'x1' set but not used}}
  x1++;
  int x2 = 0; // expected-warning{{variable 'x2' set but not used}}
  x2--;
  int x3 = 0; // expected-warning{{variable 'x3' set but not used}}
  ++x3;
  int x4 = 0; // expected-warning{{variable 'x4' set but not used}}
  --x4;

  static int counter = 0; // expected-warning{{variable 'counter' set but not used}}
  counter += 1;

  volatile int v1 = 0;
  ++v1;
  typedef volatile int volint;
  volint v2 = 0;
  v2++;
}
