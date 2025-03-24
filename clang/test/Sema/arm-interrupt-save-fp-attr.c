// RUN: %clang_cc1 %s -triple arm-apple-darwin  -target-feature +vfp2 -verify -fsyntax-only
// RUN: %clang_cc1 %s -triple thumb-apple-darwin  -target-feature +vfp3 -verify -fsyntax-only
// RUN: %clang_cc1 %s -triple armeb-none-eabi  -target-feature +vfp4 -verify -fsyntax-only
// RUN: %clang_cc1 %s -triple thumbeb-none-eabi  -target-feature +neon -verify -fsyntax-only
// RUN: %clang_cc1 %s -triple thumbeb-none-eabi -target-feature +neon -target-feature +soft-float -DSOFT -verify -fsyntax-only

#ifndef SOFT
__attribute__((interrupt_save_fp(IRQ))) void foo() {} // expected-error {{'interrupt_save_fp' attribute requires a string}}
__attribute__((interrupt_save_fp("irq"))) void foo1() {} // expected-warning {{'interrupt_save_fp' attribute argument not supported: irq}}

__attribute__((interrupt_save_fp("IRQ", 1))) void foo2() {} // expected-error {{'interrupt_save_fp' attribute takes no more than 1 argument}}
__attribute__((interrupt_save_fp("IRQ"))) void foo3() {}
__attribute__((interrupt_save_fp("FIQ"))) void foo4() {}
__attribute__((interrupt_save_fp("SWI"))) void foo5() {}
__attribute__((interrupt_save_fp("ABORT"))) void foo6() {}
__attribute__((interrupt_save_fp("UNDEF"))) void foo7() {}
__attribute__((interrupt_save_fp)) void foo8() {}
__attribute__((interrupt_save_fp())) void foo9() {}
__attribute__((interrupt_save_fp(""))) void foo10() {}
void callee1();
__attribute__((interrupt_save_fp("IRQ"))) void callee2();
void caller1() {
  callee1();
  callee2();
}
__attribute__((interrupt_save_fp("IRQ"))) void caller2() {
  callee1();
  callee2();
}

void (*callee3)();
__attribute__((interrupt_save_fp("IRQ"))) void caller3() {
  callee3();
}
#else
__attribute__((interrupt_save_fp("IRQ"))) void foo3() {} // expected-warning {{`interrupt_save_fp` only applies to targets that have a VFP unit enabled for this compilation; this will be treated as a regular `interrupt` attribute}}
__attribute__((interrupt_save_fp("FIQ"))) void foo4() {} // expected-warning {{`interrupt_save_fp` only applies to targets that have a VFP unit enabled for this compilation; this will be treated as a regular `interrupt` attribute}}
__attribute__((interrupt_save_fp("SWI"))) void foo5() {} // expected-warning {{`interrupt_save_fp` only applies to targets that have a VFP unit enabled for this compilation; this will be treated as a regular `interrupt` attribute}}
__attribute__((interrupt_save_fp("ABORT"))) void foo6() {} // expected-warning {{`interrupt_save_fp` only applies to targets that have a VFP unit enabled for this compilation; this will be treated as a regular `interrupt` attribute}}
__attribute__((interrupt_save_fp("UNDEF"))) void foo7() {} // expected-warning {{`interrupt_save_fp` only applies to targets that have a VFP unit enabled for this compilation; this will be treated as a regular `interrupt` attribute}}
__attribute__((interrupt_save_fp)) void foo8() {} // expected-warning {{`interrupt_save_fp` only applies to targets that have a VFP unit enabled for this compilation; this will be treated as a regular `interrupt` attribute}}
__attribute__((interrupt_save_fp())) void foo9() {} // expected-warning {{`interrupt_save_fp` only applies to targets that have a VFP unit enabled for this compilation; this will be treated as a regular `interrupt` attribute}}
__attribute__((interrupt_save_fp(""))) void foo10() {} // expected-warning {{`interrupt_save_fp` only applies to targets that have a VFP unit enabled for this compilation; this will be treated as a regular `interrupt` attribute}}
void callee1();
__attribute__((interrupt_save_fp("IRQ"))) void callee2(); // expected-warning {{`interrupt_save_fp` only applies to targets that have a VFP unit enabled for this compilation; this will be treated as a regular `interrupt` attribute}}
void caller1() {
  callee1();
  callee2();
}
__attribute__((interrupt_save_fp("IRQ"))) void caller2() { // expected-warning {{`interrupt_save_fp` only applies to targets that have a VFP unit enabled for this compilation; this will be treated as a regular `interrupt` attribute}}
  callee1();
  callee2();
}

void (*callee3)();
__attribute__((interrupt_save_fp("IRQ"))) void caller3() { // expected-warning {{`interrupt_save_fp` only applies to targets that have a VFP unit enabled for this compilation; this will be treated as a regular `interrupt` attribute}}
  callee3();
}
#endif