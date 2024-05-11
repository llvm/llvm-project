// RUN: %clang_cc1 %s -triple arm-none-eabi -DSOFT -verify -fsyntax-only
// RUN: %clang_cc1 %s -triple arm-none-eabi -target-feature +vfp2 -verify -fsyntax-only


#ifdef SOFT
__attribute__((interrupt("irq"))) void foo1(void) {} // expected-warning {{'interrupt' attribute argument not supported: irq}}
__attribute__((interrupt(IRQ))) void foo(void) {} // expected-error {{'interrupt' attribute requires a string}}
__attribute__((interrupt("IRQ", 1))) void foo2(void) {} // expected-error {{'interrupt' attribute takes no more than 1 argument}}
__attribute__((interrupt("IRQ"))) void foo3(void) {}
__attribute__((interrupt("FIQ"))) void foo4(void) {}
__attribute__((interrupt("SWI"))) void foo5(void) {}
__attribute__((interrupt("ABORT"))) void foo6(void) {}
__attribute__((interrupt("UNDEF"))) void foo7(void) {}
__attribute__((interrupt)) void foo8(void) {}
__attribute__((interrupt())) void foo9(void) {}
__attribute__((interrupt(""))) void foo10(void) {}
#else
__attribute__((interrupt("IRQ"))) void float_irq(void); // expected-warning {{interrupt service routine with vfp enabled may clobber the interruptee's vfp state}}
#endif
