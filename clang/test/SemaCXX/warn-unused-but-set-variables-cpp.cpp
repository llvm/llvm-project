// RUN: %clang_cc1 -fblocks -fsyntax-only -Wunused-but-set-variable -verify %s

struct S {
  int i;
};

int f0() {
  int y; // expected-warning{{variable 'y' set but not used}}
  y = 0;

  int z __attribute__((unused));
  z = 0;

  // In C++, don't warn for structs. (following gcc's behavior)
  struct S s;
  struct S t;
  s = t;

  int x;
  x = 0;
  return x + 5;
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

void f2() {
  // Don't warn for either of these cases.
  constexpr int x = 2;
  const int y = 1;
  char a[x];
  char b[y];
}
