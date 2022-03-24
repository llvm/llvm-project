// RUN: %clang_cc1 -fblocks -fsyntax-only -Wunused-but-set-variable -verify %s

struct S {
  int i;
};

struct __attribute__((warn_unused)) SWarnUnused {
  int j;
  void operator +=(int);
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

  // Unless it's marked with the warn_unused attribute.
  struct SWarnUnused swu; // expected-warning{{variable 'swu' set but not used}}
  struct SWarnUnused swu2;
  swu = swu2;

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

void f3(int n) {
  // Don't warn for overloaded compound assignment operators.
  SWarnUnused swu;
  swu += n;
}

template<typename T> void f4(T n) {
  // Don't warn for (potentially) overloaded compound assignment operators in
  // template code.
  SWarnUnused swu;
  swu += n;
}
