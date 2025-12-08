// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu %s -fclangir -fclangir-analysis="fallthrough" -emit-cir -o /dev/null -verify
// INFO: These test cases are derived from clang/test/Sema/return.c

int test1(void) {
} // expected-warning {{non-void function does not return a value}}

int test3(void) {
  goto a;
  a: ;
} // expected-warning {{non-void function does not return a value}}

int test20(void) {
  int i;
  if (i)
    return 0;
  else if (0)
    return 2;
} // expected-warning {{non-void function does not return a value in all control paths}}

int test22(void) {
  int i;
  switch (i) default: ;
} // expected-warning {{non-void function does not return a value}}

int test23(void) {
  int i;
  switch (i) {
  case 0:
    return 0;
  case 2:
    return 2;
  }
} // expected-warning {{non-void function does not return a value in all control paths}}


