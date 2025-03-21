// RUN: %clang_cc1 -triple riscv32-unknown-elf -emit-llvm -DCHECK_IR < %s| FileCheck %s
// RUN: %clang_cc1 -triple riscv64-unknown-elf -emit-llvm -DCHECK_IR < %s| FileCheck %s
// RUN: %clang_cc1 %s -triple riscv32-unknown-elf -verify -fsyntax-only
// RUN: %clang_cc1 %s -triple riscv64-unknown-elf -verify -fsyntax-only

#if defined(CHECK_IR)
// CHECK-LABEL:  @foo_stack_swap() #0
// CHECK: ret void
__attribute__((interrupt("SiFive-CLIC-stack-swap"))) void foo_stack_swap(void) {}

// CHECK-LABEL:  @foo_preemptible() #1
// CHECK: ret void
__attribute__((interrupt("SiFive-CLIC-preemptible"))) void foo_preemptible(void) {}

// CHECK-LABEL:  @foo_stack_swap_preemptible() #2
// CHECK: ret void
__attribute__((interrupt("SiFive-CLIC-stack-swap", "SiFive-CLIC-preemptible")))
void foo_stack_swap_preemptible(void) {}

// CHECK-LABEL:  @foo_preemptible_stack_swap() #2
// CHECK: ret void
__attribute__((interrupt("SiFive-CLIC-preemptible", "SiFive-CLIC-stack-swap")))
void foo_preemptible_stack_swap(void) {}

// CHECK-LABEL:  @foo_stack_swap_repeat() #0
// CHECK: ret void
__attribute__((interrupt("SiFive-CLIC-stack-swap", "SiFive-CLIC-stack-swap")))
void foo_stack_swap_repeat(void) {}

// CHECK-LABEL:  @foo_preemptible_repeat() #1
// CHECK: ret void
__attribute__((interrupt("SiFive-CLIC-preemptible", "SiFive-CLIC-preemptible")))
void foo_preemptible_repeat(void) {}

// CHECK-LABEL:  @foo_machine_stack_swap() #0
// CHECK: ret void
__attribute__((interrupt("machine", "SiFive-CLIC-stack-swap")))
void foo_machine_stack_swap(void) {}

// CHECK-LABEL:  @foo_stack_swap_machine() #0
// CHECK: ret void
__attribute__((interrupt("SiFive-CLIC-stack-swap", "machine")))
void foo_stack_swap_machine(void) {}

// CHECK-LABEL:  @foo_preemptible_machine() #1
// CHECK: ret void
__attribute__((interrupt("SiFive-CLIC-preemptible", "machine")))
void foo_preemptible_machine(void) {}

// CHECK-LABEL:  @foo_machine_preemptible() #1
// CHECK: ret void
__attribute__((interrupt("machine", "SiFive-CLIC-preemptible")))
void foo_machine_preemptible(void) {}


// CHECK: attributes #0
// CHECK: "interrupt"="SiFive-CLIC-stack-swap"
// CHECK: attributes #1
// CHECK: "interrupt"="SiFive-CLIC-preemptible"
// CHECK: attributes #2
// CHECK: "interrupt"="SiFive-CLIC-preemptible-stack-swap"
#else

__attribute__((interrupt("SiFive-CLIC-stack-swap"))) void foo15(void);
__attribute__((interrupt("SiFive-CLIC-stack-swap", "SiFive-CLIC-stack-swap"))) void foo15(void);
__attribute__((interrupt("SiFive-CLIC-stack-swap", "machine"))) void foo15(void);
__attribute__((interrupt("machine", "SiFive-CLIC-stack-swap"))) void foo15(void);

__attribute__((interrupt("SiFive-CLIC-preemptible"))) void foo15(void);
__attribute__((interrupt("SiFive-CLIC-preemptible", "SiFive-CLIC-preemptible"))) void foo15(void);
__attribute__((interrupt("SiFive-CLIC-preemptible", "machine"))) void foo15(void);
__attribute__((interrupt("machine", "SiFive-CLIC-preemptible"))) void foo15(void);

__attribute__((interrupt("SiFive-CLIC-stack-swap"))) void foo16(void) {}
__attribute__((interrupt("SiFive-CLIC-stack-swap", "SiFive-CLIC-stack-swap"))) void foo17(void) {}
__attribute__((interrupt("SiFive-CLIC-stack-swap", "machine"))) void foo18(void) {}
__attribute__((interrupt("machine", "SiFive-CLIC-stack-swap"))) void foo19(void) {}

__attribute__((interrupt("SiFive-CLIC-preemptible"))) void foo20(void) {}
__attribute__((interrupt("SiFive-CLIC-preemptible", "SiFive-CLIC-preemptible"))) void foo21(void) {}
__attribute__((interrupt("SiFive-CLIC-preemptible", "machine"))) void foo22(void) {}
__attribute__((interrupt("machine", "SiFive-CLIC-preemptible"))) void foo23(void) {}

__attribute__((interrupt("machine", "machine", "SiFive-CLIC-preemptible"))) void foo24(void) {} // expected-error {{'interrupt' attribute takes no more than 2 arguments}}

__attribute__((interrupt("SiFive-CLIC-preemptible", "supervisor"))) void foo27(void) {} // expected-error {{RISC-V 'interrupt' attribute contains invalid combination of interrupt types}}

__attribute__((interrupt("supervisor", "SiFive-CLIC-stack-swap"))) void foo28(void) {} // expected-error {{RISC-V 'interrupt' attribute contains invalid combination of interrupt types}}

__attribute__((interrupt("SiFive-CLIC-stack-swap", 1))) void foo29(void) {} // expected-error {{expected string literal as argument of 'interrupt' attribute}}

__attribute__((interrupt(1, "SiFive-CLIC-stack-swap"))) void foo30(void) {} // expected-error {{expected string literal as argument of 'interrupt' attribute}}

__attribute__((interrupt("SiFive-CLIC-stack-swap", "foo"))) void foo31(void) {} // expected-warning {{'interrupt' attribute argument not supported: "foo"}}

__attribute__((interrupt("foo", "SiFive-CLIC-stack-swap"))) void foo32(void) {} // expected-warning {{'interrupt' attribute argument not supported: "foo"}}

#endif
