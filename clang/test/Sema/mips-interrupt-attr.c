// RUN: %clang_cc1 %s -triple mips-img-elf -verify -fsyntax-only
struct a { int b; };

struct a test __attribute__((interrupt)); // expected-warning {{'interrupt' attribute only applies to functions and methods}}

__attribute__((interrupt("EIC"))) void foo1() {} // expected-warning {{'interrupt' attribute argument not supported: 'EIC'}}

__attribute__((interrupt("eic", 1))) void foo2() {} // expected-error {{'interrupt' attribute takes no more than 1 argument}}

__attribute__((interrupt("eic"))) void foo3() {}
__attribute__((interrupt("vector=sw0"))) void foo4() {}
__attribute__((interrupt("vector=hw0"))) void foo5() {}
__attribute__((interrupt("vector=hw1"))) void foo6() {}
__attribute__((interrupt("vector=hw2"))) void foo7() {}
__attribute__((interrupt("vector=hw3"))) void foo8() {}
__attribute__((interrupt("vector=hw4"))) void foo9() {}
__attribute__((interrupt("vector=hw5"))) void fooa() {}
__attribute__((interrupt(""))) void food() {}

__attribute__((interrupt)) int foob() {return 0;} // expected-warning {{MIPS 'interrupt' attribute only applies to functions that have a 'void' return type}}
__attribute__((interrupt())) void fooc(int a) {} // expected-warning {{MIPS 'interrupt' attribute only applies to functions that have no parameters}}
__attribute__((interrupt,mips16)) void fooe() {} // expected-error {{'interrupt' and 'mips16' attributes are not compatible}} \
                                                 // expected-note {{conflicting attribute is here}}
__attribute__((mips16,interrupt)) void foof() {} // expected-error {{'mips16' and 'interrupt' attributes are not compatible}} \
                                                 // expected-note {{conflicting attribute is here}}
__attribute__((interrupt)) __attribute__ ((mips16)) void foo10() {} // expected-error {{'interrupt' and 'mips16' attributes are not compatible}} \
                                                                    // expected-note {{conflicting attribute is here}}
__attribute__((mips16)) __attribute ((interrupt)) void foo11() {} // expected-error {{'mips16' and 'interrupt' attributes are not compatible}} \
                                                                  // expected-note {{conflicting attribute is here}}
