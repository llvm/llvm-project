// RUN: %clang_cc1 %s -Winteger-overflow -Wno-unused-value -foverflow-behavior-types -Woverflow-behavior-conversion -verify -fsyntax-only

typedef int __attribute__((overflow_behavior)) bad_arg_count; // expected-error {{'overflow_behavior' attribute takes one argument}}
typedef int __attribute__((overflow_behavior(not_real))) bad_arg_spec; // expected-error {{'not_real' is not a valid argument to attribute 'overflow_behavior'}}
typedef int __attribute__((overflow_behavior("not_real"))) bad_arg_spec_str; // expected-error {{'not_real' is not a valid argument to attribute 'overflow_behavior'}}
typedef char* __attribute__((overflow_behavior("wrap"))) bad_type; // expected-warning {{'overflow_behavior' attribute cannot be applied to non-integer type 'char *'; attribute ignored}}

typedef int __attribute__((overflow_behavior(wrap))) ok_wrap; // OK
typedef long __attribute__((overflow_behavior(no_wrap))) ok_nowrap; // OK
typedef unsigned long __attribute__((overflow_behavior("wrap"))) str_ok_wrap; // OK
typedef char __attribute__((overflow_behavior("no_wrap"))) str_ok_nowrap; // OK

void foo() {
  (2147483647 + 100); // expected-warning {{overflow in expression; result is }}
  (ok_wrap)2147483647 + 100; // no warn
}

#define __no_wrap __attribute__((overflow_behavior(no_wrap)))

void ptr(int a) {
  int __no_wrap *p = &a; // expected-warning-re {{incompatible pointer types initializing '__no_wrap int *' {{.*}}of type 'int *'}}
}

void ptr2(__no_wrap int a) {
  int *p = &a; // expected-warning-re {{incompatible pointer types initializing 'int *' {{.*}}of type '__no_wrap int *'}}
}


// verify semantics of -Wimplicitly-discarded-overflow-behavior{,-pedantic}
void imp_disc_pedantic(unsigned a) {}
void imp_disc(int a) {}
void imp_disc_test(unsigned __attribute__((overflow_behavior(wrap))) a) {
  imp_disc_pedantic(a); // expected-warning {{implicit conversion from '__wrap unsigned int' to 'unsigned int' discards overflow behavior}}
  imp_disc(a); // expected-warning {{implicit conversion from '__wrap unsigned int' to 'int' discards overflow behavior}}
}

// -Wconversion for assignments that discard overflow behavior
void assignment_disc_test(unsigned __attribute__((overflow_behavior(wrap))) a) {
  int b = a; // expected-warning {{implicit conversion from '__wrap unsigned int' to 'int' during assignment discards overflow behavior}}
  int c = (int)a; // OK
}
