// RUN: %clang_cc1 -triple riscv64-unknown-elf \
// RUN:   -target-feature +experimental-smnip \
// RUN:   -target-feature +experimental-ssnip \
// RUN:   -target-feature +experimental-smip \
// RUN:   -target-feature +experimental-ssip -emit-llvm -DCHECK_IR %s -o - \
// RUN:   | FileCheck %s
// RUN: %clang_cc1 -triple riscv64-unknown-elf \
// RUN:   -target-feature +experimental-smnip \
// RUN:   -target-feature +experimental-ssnip \
// RUN:   -target-feature +experimental-smip \
// RUN:   -target-feature +experimental-ssip -verify=enabled,both \
// RUN:   -fsyntax-only %s
// RUN: %clang_cc1 -triple riscv64-unknown-elf -verify=disabled,both \
// RUN:   -fsyntax-only %s

#ifdef CHECK_IR
__attribute__((interrupt("preemptible"))) void m_preemptible(void) {}
__attribute__((interrupt("supervisor", "preemptible")))
void s_preemptible(void) {}
__attribute__((interrupt("push-pop"))) void m_push_pop(void) {}
__attribute__((interrupt("supervisor", "push-pop"))) void s_push_pop(void) {}
__attribute__((interrupt("machine", "preemptible", "push-pop")))
void m_combined(void) {}
__attribute__((interrupt("push-pop", "supervisor", "preemptible")))
void s_combined(void) {}

// CHECK: attributes #0 = {{.*}}"interrupt"="machine-preemptible"
// CHECK: attributes #1 = {{.*}}"interrupt"="supervisor-preemptible"
// CHECK: attributes #2 = {{.*}}"interrupt"="machine-push-pop"
// CHECK: attributes #3 = {{.*}}"interrupt"="supervisor-push-pop"
// CHECK: attributes #4 = {{.*}}"interrupt"="machine-preemptible-push-pop"
// CHECK: attributes #5 = {{.*}}"interrupt"="supervisor-preemptible-push-pop"
#else
__attribute__((interrupt("preemptible"))) void m_preemptible(void) {} // disabled-error {{RISC-V 'interrupt' attribute 'preemptible' requires extension 'Smnip'}}
__attribute__((interrupt("supervisor", "preemptible"))) void s_preemptible(void) {} // disabled-error {{RISC-V 'interrupt' attribute 'preemptible' requires extension 'Ssnip'}}
__attribute__((interrupt("push-pop"))) void m_push_pop(void) {} // disabled-error {{RISC-V 'interrupt' attribute 'push-pop' requires extension 'Smip'}}
__attribute__((interrupt("supervisor", "push-pop"))) void s_push_pop(void) {} // disabled-error {{RISC-V 'interrupt' attribute 'push-pop' requires extension 'Ssip'}}

__attribute__((interrupt("machine", "supervisor", "preemptible"))) void bad_modes(void) {} // both-error {{RISC-V 'interrupt' attribute contains invalid combination of interrupt types}}
__attribute__((interrupt("preemptible", "SiFive-CLIC-preemptible"))) void bad_sifive(void) {} // both-error {{RISC-V 'interrupt' attribute contains invalid combination of interrupt types}}
__attribute__((interrupt("push-pop", "rnmi"))) void bad_rnmi(void) {} // both-error {{RISC-V 'interrupt' attribute contains invalid combination of interrupt types}}
__attribute__((interrupt("preemptible", "qci-nest"))) void bad_qci(void) {} // both-error {{RISC-V 'interrupt' attribute contains invalid combination of interrupt types}}
#endif
