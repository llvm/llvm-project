// RUN: %clang_cc1 -fsyntax-only %s -verify
// expected-no-diagnostics

struct A {
  int i;
};
struct B {
  struct A *a;
};
const struct B c = {&(struct A){1}};

int main(void) {
  if ((c.a->i != 1) || (c.a->i)) {
    return 1;
  }
  return 0;
}
