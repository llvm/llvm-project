// RUN: %clang_cc1 -fsyntax-only -verify %s

struct c {int x;};
int a(struct c x, long long y) {
  void const* l1_ptr = &&l1;
  goto *l1_ptr;
l1:
  goto *x; // expected-error{{incompatible type}}
  goto *y; // expected-error{{incompatible integer to pointer conversion}}
}

