// RUN: %clang_cc1 %s -triple arm-none-eabi -verify -fsyntax-only
// RUN: %clang_cc1 %s -triple arm-none-eabi -target-feature +vfp2 -verify -fsyntax-only


#if !defined(__ARM_FP)
__attribute__((interrupt_save_fp("IRQ"))) void float_irq(void); // expected-warning {{`interrupt_save_fp` only applies to targets that have a VFP unit enabled for this compilation; this will be treated as a regular `interrupt` attribute}}
#else // defined(__ARM_FP)
__attribute__((interrupt_save_fp("irq"))) void foo1(void) {} // expected-warning {{'interrupt_save_fp' attribute argument not supported: irq}}
__attribute__((interrupt_save_fp(IRQ))) void foo(void) {} // expected-error {{'interrupt_save_fp' attribute requires a string}}
__attribute__((interrupt_save_fp("IRQ", 1))) void foo2(void) {} // expected-error {{'interrupt_save_fp' attribute takes no more than 1 argument}}
__attribute__((interrupt_save_fp("IRQ"))) void foo3(void) {}
__attribute__((interrupt_save_fp("FIQ"))) void foo4(void) {}
__attribute__((interrupt_save_fp("SWI"))) void foo5(void) {}
__attribute__((interrupt_save_fp("ABORT"))) void foo6(void) {}
__attribute__((interrupt_save_fp("UNDEF"))) void foo7(void) {}
__attribute__((interrupt_save_fp)) void foo8(void) {}
__attribute__((interrupt_save_fp())) void foo9(void) {}
__attribute__((interrupt_save_fp(""))) void foo10(void) {}

__attribute__((interrupt_save_fp("IRQ"))) void callee(void) {}

void caller(void)
{
    callee(); // expected-error {{interrupt service routine cannot be called directly}}
}
#endif // __ARM_FP
