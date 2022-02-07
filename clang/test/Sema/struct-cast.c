// RUN: %clang_cc1 -fsyntax-only %s -verify
// expected-no-diagnostics

struct S {
 int one;
 int two;
};

struct S const foo(void);


struct S tmp;

void priv_sock_init(void) {
  tmp = (struct S)foo();
}
