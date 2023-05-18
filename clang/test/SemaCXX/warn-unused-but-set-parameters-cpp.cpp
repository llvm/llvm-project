// RUN: %clang_cc1 -fblocks -fsyntax-only -Wunused-but-set-parameter -verify %s

int f0(int x,
       int y, // expected-warning{{parameter 'y' set but not used}}
       int z __attribute__((unused))) {
  y = 0;
  return x;
}

void f1(void) {
  (void)^(int x,
          int y, // expected-warning{{parameter 'y' set but not used}}
          int z __attribute__((unused))) {
    y = 0;
    return x;
  };
}

struct S {
  int i;
};

// In C++, don't warn for a struct (following gcc).
void f3(struct S s) {
  struct S t;
  s = t;
}

// Also don't warn for a reference.
void f4(int &x) {
  x = 0;
}

// Make sure this doesn't warn.
struct A {
  int i;
  A(int j) : i(j) {}
};
