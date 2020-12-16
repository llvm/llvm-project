// RUN: %clang_cc1 -verify -fsyntax-only %s

void f1() __attribute__((leaf));

void f2() __attribute__((leaf("abc"))); // expected-error {{'leaf' attribute takes no argument}}

int var __attribute__ ((leaf())); // expected-warning {{'leaf' attribute only applies to functions}}

// FIXME: Might diagnose a warning if leaf attribute is used in function definition
// The leaf attribute has no effect on functions defined within the current compilation unit
__attribute__((leaf)) void f3() {
}
