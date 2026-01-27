// RUN: %clang_cc1 -triple x86_64-unknown-unknown -std=c++17 -ftrivial-auto-var-init=zero -Wtrivial-auto-var-init -emit-llvm -o /dev/null -verify %s

void use(int *);
void use(void *);

struct Trivial {
  int a;
  int b;
};

void uninitialized_attr_precase(int c) {
  switch (c) {
    [[clang::uninitialized]] int x; // no warning
  case 0:
    x = 1;
    use(&x);
    break;
  }
}

void struct_precase(int c) {
  switch (c) {
    Trivial t; // expected-warning{{'t' cannot be initialized with '-ftrivial-auto-var-init'}}
  case 0:
    t.a = 1;
    use(&t);
    break;
  }
}

void int_precase(int c) {
  switch (c) {
    int x; // expected-warning{{'x' cannot be initialized with '-ftrivial-auto-var-init'}}
  case 0:
    x = 1;
    use(&x);
    break;
  }
}
