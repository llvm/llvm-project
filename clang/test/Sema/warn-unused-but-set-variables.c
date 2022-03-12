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
