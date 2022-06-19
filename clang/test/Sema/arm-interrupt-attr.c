// RUN: %clang_cc1 %s -triple arm-apple-darwin  -target-feature +vfp2 -verify -fsyntax-only
// RUN: %clang_cc1 %s -triple thumb-apple-darwin  -target-feature +vfp3 -verify -fsyntax-only
// RUN: %clang_cc1 %s -triple armeb-none-eabi  -target-feature +vfp4 -verify -fsyntax-only
// RUN: %clang_cc1 %s -triple thumbeb-none-eabi  -target-feature +neon -verify -fsyntax-only
// RUN: %clang_cc1 %s -triple thumbeb-none-eabi -target-feature +neon -target-feature +soft-float -DSOFT -verify -fsyntax-only

__attribute__((interrupt(IRQ))) void foo(void) {} // expected-error {{'interrupt' attribute requires a string}}
__attribute__((interrupt("irq"))) void foo1(void) {} // expected-warning {{'interrupt' attribute argument not supported: irq}}

__attribute__((interrupt("IRQ", 1))) void foo2(void) {} // expected-error {{'interrupt' attribute takes no more than 1 argument}}

__attribute__((interrupt("IRQ"))) void foo3(void) {}
__attribute__((interrupt("FIQ"))) void foo4(void) {}
__attribute__((interrupt("SWI"))) void foo5(void) {}
__attribute__((interrupt("ABORT"))) void foo6(void) {}
__attribute__((interrupt("UNDEF"))) void foo7(void) {}

__attribute__((interrupt)) void foo8(void) {}
__attribute__((interrupt())) void foo9(void) {}
__attribute__((interrupt(""))) void foo10(void) {}

#ifndef SOFT
// expected-note@+2 {{'callee1' declared here}}
#endif
void callee1(void);
__attribute__((interrupt("IRQ"))) void callee2(void);
void caller1(void) {
  callee1();
  callee2();
}

#ifndef SOFT
__attribute__((interrupt("IRQ"))) void caller2(void) {
  callee1(); // expected-warning {{call to function without interrupt attribute could clobber interruptee's VFP registers}}
  callee2();
}

void (*callee3)(void);
__attribute__((interrupt("IRQ"))) void caller3(void) {
  callee3(); // expected-warning {{call to function without interrupt attribute could clobber interruptee's VFP registers}}
}
#else
__attribute__((interrupt("IRQ"))) void caller2(void) {
  callee1();
  callee2();
}

void (*callee3)(void);
__attribute__((interrupt("IRQ"))) void caller3(void) {
  callee3();
}
#endif
