// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu %s -fclangir -fclangir-analysis="fallthrough" -emit-cir -o /dev/null -verify
// INFO: These test cases are derived from clang/test/Sema/return.c
int unknown(void);

int test7(void) {
  unknown();
} // expected-warning {{non-void function does not return a value}}

int test8(void) {
  (void)(1 + unknown());
} // expected-warning {{non-void function does not return a value}}



int test14(void) {
  (void)(1 || unknown());
} // expected-warning {{non-void function does not return a value}}

int test15(void) {
  (void)(0 || unknown());
} // expected-warning {{non-void function does not return a value}}

int test16(void) {
  (void)(0 && unknown());
} // expected-warning {{non-void function does not return a value}}

int test17(void) {
  (void)(1 && unknown());
} // expected-warning {{non-void function does not return a value}}


