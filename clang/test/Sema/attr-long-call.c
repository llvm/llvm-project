// RUN: %clang_cc1 -triple mips-linux-gnu -fsyntax-only -verify %s
// RUN: %clang_cc1 -triple mips64-linux-gnu -fsyntax-only -verify %s

__attribute__((long_call(0))) void foo1(void);  // expected-error {{'long_call' attribute takes no arguments}}
__attribute__((short_call(0))) void foo9(void);  // expected-error {{'short_call' attribute takes no arguments}}
__attribute__((far(0))) void foo2(void);  // expected-error {{'far' attribute takes no arguments}}
__attribute__((near(0))) void foo3(void);  // expected-error {{'near' attribute takes no arguments}}

__attribute((long_call)) int a; // expected-warning {{attribute only applies to functions}}
__attribute((short_call)) int d; // expected-warning {{attribute only applies to functions}}
__attribute((far)) int a; // expected-warning {{attribute only applies to functions}}
__attribute((near)) int a; // expected-warning {{attribute only applies to functions}}

__attribute((long_call)) void foo4(void);
__attribute((short_call)) void foo10(void);
__attribute((far)) void foo5(void);
__attribute((near)) void foo6(void);

__attribute((long_call, far)) void foo7(void);
__attribute((short_call, near)) void foo11(void);

__attribute((far, near)) void foo8(void); // expected-error {{'near' and 'far' attributes are not compatible}} \
                                      // expected-note {{conflicting attribute is here}}

__attribute((short_call, long_call)) void foo12(void); // expected-error {{'long_call' and 'short_call' attributes are not compatible}} \
                                                   // expected-note {{conflicting attribute is here}}
