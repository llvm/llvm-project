// RUN: %clang_cc1 -triple mips-linux-gnu -fsyntax-only -verify %s

__attribute__((long_call(0))) void foo1();  // expected-error {{'long_call' attribute takes no arguments}}
__attribute__((far(0))) void foo2();  // expected-error {{'far' attribute takes no arguments}}
__attribute__((near(0))) void foo3();  // expected-error {{'near' attribute takes no arguments}}

__attribute((long_call)) int a; // expected-warning {{attribute only applies to functions}}
__attribute((far)) int a; // expected-warning {{attribute only applies to functions}}
__attribute((near)) int a; // expected-warning {{attribute only applies to functions}}

__attribute((long_call)) void foo4();
__attribute((far)) void foo5();
__attribute((near)) void foo6();

__attribute((long_call, far)) void foo7();

__attribute((far, near)) void foo8(); // expected-error {{'far' and 'near' attributes are not compatible}} \
                                      // expected-note {{conflicting attribute is here}}
