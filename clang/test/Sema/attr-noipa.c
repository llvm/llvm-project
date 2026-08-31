// RUN: %clang_cc1 -verify -fsyntax-only %s
// RUN: %clang_cc1 -verify -fsyntax-only -x c++ %s

__attribute__((noipa)) int a; // expected-warning {{'noipa' attribute only applies to functions}}

__attribute__((noipa)) void t1(void);

__attribute__((noipa(2))) void t2(void); // expected-error {{'noipa' attribute takes no arguments}}
