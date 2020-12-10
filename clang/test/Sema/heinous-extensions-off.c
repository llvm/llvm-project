// RUN: %clang_cc1 %s -verify

int foo() {
        int a;
        // PR3788
        asm("nop" : : "m"((int)(a))); // expected-error {{cast in a inline asm context requiring an lvalue}}
        // PR3794
        asm("nop" : "=r"((unsigned)a)); // expected-error {{cast in a inline asm context requiring an lvalue}}
}

