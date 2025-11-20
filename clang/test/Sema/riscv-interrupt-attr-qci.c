// RUN: %clang_cc1 -triple riscv32-unknown-elf -target-feature +experimental-xqciint -emit-llvm -DCHECK_IR < %s | FileCheck %s
// RUN: %clang_cc1 %s -triple riscv32-unknown-elf -target-feature +experimental-xqciint -verify=enabled,both -fsyntax-only
// RUN: %clang_cc1 %s -triple riscv32-unknown-elf -verify=disabled,both -fsyntax-only
// RUN: %clang_cc1 %s -triple riscv32-unknown-elf -target-feature -experimental-xqciint -verify=disabled,both -fsyntax-only
// RUN: %clang_cc1 %s -triple riscv64-unknown-elf -verify=disabled,both -fsyntax-only -DRV64

#if defined(CHECK_IR)
// Test for QCI extension's interrupt attribute support
// CHECK-LABEL: @foo_nest_interrupt() #0
// CHECK: ret void
__attribute__((interrupt("qci-nest")))
void foo_nest_interrupt(void) {}

// CHECK-LABEL: @foo_nest_nest_interrupt() #0
// CHECK: ret void
__attribute__((interrupt("qci-nest", "qci-nest")))
void foo_nest_nest_interrupt(void) {}

// CHECK-LABEL: @foo_nonest_interrupt() #1
// CHECK: ret void
__attribute__((interrupt("qci-nonest")))
void foo_nonest_interrupt(void) {}

// CHECK-LABEL: @foo_nonest_nonest_interrupt() #1
// CHECK: ret void
__attribute__((interrupt("qci-nonest", "qci-nonest")))
void foo_nonest_nonest_interrupt(void) {}

// CHECK: attributes #0
// CHECK: "interrupt"="qci-nest"
// CHECK: attributes #1
// CHECK: "interrupt"="qci-nonest"
#else
// Test for QCI extension's interrupt attribute support
__attribute__((interrupt(1))) void foo1(void) {} // both-error {{expected string literal as argument of 'interrupt' attribute}}
__attribute__((interrupt("qci-nonest", 1))) void foo_nonest2(void) {} // both-error {{expected string literal as argument of 'interrupt' attribute}}
__attribute__((interrupt("qci-nest", 1))) void foo_nest2(void) {} // both-error {{expected string literal as argument of 'interrupt' attribute}}
__attribute__((interrupt("qci-est"))) void foo_nest3(void) {} // both-warning {{'interrupt' attribute argument not supported: "qci-est"}}
__attribute__((interrupt("qci-noest"))) void foo_nonest3(void) {} // both-warning {{'interrupt' attribute argument not supported: "qci-noest"}}
__attribute__((interrupt("", "qci-nonest"))) void foo_nonest4(void) {} // both-warning {{'interrupt' attribute argument not supported: ""}}
__attribute__((interrupt("", "qci-nest"))) void foo_nest4(void) {} // both-warning {{'interrupt' attribute argument not supported: ""}}

__attribute__((interrupt("qci-nonest", "qci-nest"))) void foo_nonest5(void) {} // both-error {{RISC-V 'interrupt' attribute contains invalid combination of interrupt types}}
__attribute__((interrupt("qci-nest", "qci-nonest"))) void foo_nest5(void) {} // both-error {{RISC-V 'interrupt' attribute contains invalid combination of interrupt types}}

__attribute__((interrupt("qci-nest"))) void foo_nest(void) {} // disabled-error {{RISC-V 'interrupt' attribute 'qci-nest' requires extension 'Xqciint'}}
__attribute__((interrupt("qci-nonest"))) void foo_nonest(void) {} // disabled-error {{RISC-V 'interrupt' attribute 'qci-nonest' requires extension 'Xqciint'}}

__attribute__((interrupt("qci-nest", "qci-nest"))) void foo_nest_nest(void) {} // disabled-error {{RISC-V 'interrupt' attribute 'qci-nest' requires extension 'Xqciint'}}
__attribute__((interrupt("qci-nonest", "qci-nonest"))) void foo_nonest_nonest(void) {} // disabled-error {{RISC-V 'interrupt' attribute 'qci-nonest' requires extension 'Xqciint'}}


// This tests the errors for the qci interrupts when using
// `__attribute__((target(...)))` - but they fail on RV64, because you cannot
// enable xqciint on rv64.
#if __riscv_xlen == 32
__attribute__((target("arch=+xqciint"))) __attribute__((interrupt("qci-nest"))) void foo_nest_xqciint(void) {}
__attribute__((target("arch=+xqciint"))) __attribute__((interrupt("qci-nonest"))) void foo_nonest_xqciint(void) {}

// The attribute order is important, the interrupt attribute must come after the
// target attribute
__attribute__((interrupt("qci-nest"))) __attribute__((target("arch=+xqciint"))) void foo_nest_xqciint2(void) {}  // disabled-error {{RISC-V 'interrupt' attribute 'qci-nest' requires extension 'Xqciint'}}
__attribute__((interrupt("qci-nonest"))) __attribute__((target("arch=+xqciint"))) void foo_nonest_xqciint2(void) {}  // disabled-error {{RISC-V 'interrupt' attribute 'qci-nonest' requires extension 'Xqciint'}}
#endif

#endif
