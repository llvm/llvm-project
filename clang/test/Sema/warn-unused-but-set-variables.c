// RUN: %clang_cc1 -fblocks -fsyntax-only -Wunused-but-set-variable -verify %s

struct S {
  int i;
};

int f0() {
  int y; // expected-warning{{variable 'y' set but not used}}
  y = 0;

  int z __attribute__((unused));
  z = 0;

  struct S s; // expected-warning{{variable 's' set but not used}}
  struct S t;
  s = t;

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
