// RUN: %clang_cc1 -triple thumb-apple-darwin -target-abi aapcs -target-cpu cortex-m3 -emit-llvm -o - %s | FileCheck %s
// RUN: %clang_cc1 -triple arm-apple-darwin -target-abi apcs-gnu -emit-llvm -o - %s | FileCheck %s --check-prefix=CHECK-APCS

__attribute__((interrupt)) void test_generic_interrupt(void) {
  // CHECK: define{{.*}} arm_aapcscc void @test_generic_interrupt() [[GENERIC_ATTR:#[0-9]+]]

  // CHECK-APCS: define{{.*}} void @test_generic_interrupt() [[GENERIC_ATTR:#[0-9]+]]
}

__attribute__((interrupt("IRQ"))) void test_irq_interrupt(void) {
  // CHECK: define{{.*}} arm_aapcscc void @test_irq_interrupt() [[IRQ_ATTR:#[0-9]+]]
}

__attribute__((interrupt("FIQ"))) void test_fiq_interrupt(void) {
  // CHECK: define{{.*}} arm_aapcscc void @test_fiq_interrupt() [[FIQ_ATTR:#[0-9]+]]
}

__attribute__((interrupt("SWI"))) void test_swi_interrupt(void) {
  // CHECK: define{{.*}} arm_aapcscc void @test_swi_interrupt() [[SWI_ATTR:#[0-9]+]]
}

__attribute__((interrupt("ABORT"))) void test_abort_interrupt(void) {
  // CHECK: define{{.*}} arm_aapcscc void @test_abort_interrupt() [[ABORT_ATTR:#[0-9]+]]
}


__attribute__((interrupt("UNDEF"))) void test_undef_interrupt(void) {
  // CHECK: define{{.*}} arm_aapcscc void @test_undef_interrupt() [[UNDEF_ATTR:#[0-9]+]]
}

// CHECK: attributes [[GENERIC_ATTR]] = { {{.*}} {{"interrupt"[^=]}}
// CHECK: attributes [[IRQ_ATTR]] = { {{.*}} "interrupt"="IRQ"
// CHECK: attributes [[FIQ_ATTR]] = { {{.*}} "interrupt"="FIQ"
// CHECK: attributes [[SWI_ATTR]] = { {{.*}} "interrupt"="SWI"
// CHECK: attributes [[ABORT_ATTR]] = { {{.*}} "interrupt"="ABORT"
// CHECK: attributes [[UNDEF_ATTR]] = { {{.*}} "interrupt"="UNDEF"

// CHECK-APCS: attributes [[GENERIC_ATTR]] = { {{.*}} "interrupt"
