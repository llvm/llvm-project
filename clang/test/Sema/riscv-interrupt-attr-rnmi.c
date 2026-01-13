// RUN: %clang_cc1 -triple riscv32-unknown-elf -target-feature +smrnmi -emit-llvm -DCHECK_IR < %s | FileCheck %s
// RUN: %clang_cc1 %s -triple riscv32-unknown-elf -target-feature +smrnmi -verify=enabled,both -fsyntax-only
// RUN: %clang_cc1 %s -triple riscv32-unknown-elf -verify=disabled,both -fsyntax-only
// RUN: %clang_cc1 %s -triple riscv32-unknown-elf -target-feature -srnmi -verify=disabled,both -fsyntax-only

#if defined(CHECK_IR)
// CHECK-LABEL: foo_rnmi_interrupt() #0
// CHECK: ret void
__attribute__((interrupt("rnmi")))
void foo_rnmi_interrupt(void) {}

// CHECK-LABEL: @foo_rnmi_rnmi_interrupt() #0
// CHECK: ret void
__attribute__((interrupt("rnmi", "rnmi")))
void foo_rnmi_rnmi_interrupt(void) {}

// CHECK: attributes #0
// CHECK: "interrupt"="rnmi"
#else

__attribute__((interrupt("rnmi"))) void test_rnmi(void) {} // disabled-error {{RISC-V 'interrupt' attribute 'rnmi' requires extension 'Smrnmi'}}
__attribute__((interrupt("rnmi", "rnmi"))) void test_rnmi_rnmi(void) {} // disabled-error {{RISC-V 'interrupt' attribute 'rnmi' requires extension 'Smrnmi'}}

__attribute__((interrupt("rnmi", "supervisor"))) void foo_rnmi_supervisor(void) {}  // both-error {{RISC-V 'interrupt' attribute contains invalid combination of interrupt types}}
__attribute__((interrupt("rnmi", "machine"))) void foo_rnmi_machine(void) {}  // both-error {{RISC-V 'interrupt' attribute contains invalid combination of interrupt types}}

__attribute__((interrupt("RNMI"))) void test_RNMI(void) {} // both-warning {{'interrupt' attribute argument not supported: "RNMI"}}
#endif
