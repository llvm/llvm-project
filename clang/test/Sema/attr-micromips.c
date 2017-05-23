// RUN: %clang_cc1 -triple mips-linux-gnu -fsyntax-only -verify %s

__attribute__((nomicromips(0))) void foo1();  // expected-error {{'nomicromips' attribute takes no arguments}}
__attribute__((micromips(1))) void foo2();    // expected-error {{'micromips' attribute takes no arguments}}

__attribute((nomicromips)) int a; // expected-error {{attribute only applies to functions}}
__attribute((micromips)) int b;   // expected-error {{attribute only applies to functions}}

__attribute__((micromips,mips16)) void foo5();  // expected-error {{'micromips' and 'mips16' attributes are not compatible}} \
                                                // expected-note {{conflicting attribute is here}}
__attribute__((mips16,micromips)) void foo6();  // expected-error {{'mips16' and 'micromips' attributes are not compatible}} \
                                                // expected-note {{conflicting attribute is here}}

__attribute((micromips)) void foo7();
__attribute((nomicromips)) void foo8();
__attribute__((mips16)) void foo9(void) __attribute__((micromips)); // expected-error {{'micromips' and 'mips16' attributes are not compatible}} \
                                                                    // expected-note {{conflicting attribute is here}}
