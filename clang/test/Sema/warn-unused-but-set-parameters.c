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

void f3(struct S s) { // expected-warning{{parameter 's' set but not used}}
  struct S t;
  s = t;
}

void f4(int j) { // expected-warning{{parameter 'j' set but not used}}
    j++;
}

void f5(int k) { // expected-warning{{parameter 'k' set but not used}}
    --k;
}
