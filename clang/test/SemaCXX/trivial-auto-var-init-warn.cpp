// RUN: %clang_cc1 -std=c++17 -ftrivial-auto-var-init=zero -Wtrivial-auto-var-init -fsyntax-only -verify %s

void use(int *);
void use(void *);

struct Trivial {
  int a, b;
};

void uninitialized_attr(int c) {
  switch (c) {
    [[clang::uninitialized]] int x;
  case 0:
    x = 1;
    use(&x);
    break;
  }
}

void struct_precase(int c) {
  switch (c) {
    Trivial t; // expected-warning{{variable 't' is uninitialized and cannot be initialized with '-ftrivial-auto-var-init' because it is unreachable}}
  case 0:
    t.a = 1;
    use(&t);
    break;
  }
}

void int_precase(int c) {
  switch (c) {
    int x; // expected-warning{{variable 'x' is uninitialized and cannot be initialized with '-ftrivial-auto-var-init' because it is unreachable}}
  case 0:
    x = 1;
    use(&x);
    break;
  }
}
