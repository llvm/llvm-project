// RUN: not %clang_cc1 -S %s -verify -o -

void foo(void) {
    extern long bar[];
    asm ("" : "=r"(bar)); // expected-error{{output size should not be zero}}
}
