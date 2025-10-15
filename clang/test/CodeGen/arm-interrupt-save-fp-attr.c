// RUN: %clang_cc1 -triple thumb-apple-darwin -target-abi aapcs -target-feature +vfp4 -target-cpu cortex-m3 -emit-llvm -o - %s | FileCheck %s
// RUN: %clang_cc1 -triple arm-apple-darwin -target-abi apcs-gnu -target-feature +vfp4 -emit-llvm -o - %s | FileCheck %s --check-prefix=CHECK-APCS

__attribute__((interrupt_save_fp)) void test_generic_interrupt() {
  // CHECK: define{{.*}} arm_aapcscc void @test_generic_interrupt() [[GENERIC_ATTR:#[0-9]+]]

  // CHECK-APCS: define{{.*}} void @test_generic_interrupt() [[GENERIC_ATTR:#[0-9]+]]
}

__attribute__((interrupt_save_fp("IRQ"))) void test_irq_interrupt() {
  // CHECK: define{{.*}} arm_aapcscc void @test_irq_interrupt() [[IRQ_ATTR:#[0-9]+]]
}

__attribute__((interrupt_save_fp("FIQ"))) void test_fiq_interrupt() {
  // CHECK: define{{.*}} arm_aapcscc void @test_fiq_interrupt() [[FIQ_ATTR:#[0-9]+]]
}

__attribute__((interrupt_save_fp("SWI"))) void test_swi_interrupt() {
  // CHECK: define{{.*}} arm_aapcscc void @test_swi_interrupt() [[SWI_ATTR:#[0-9]+]]
}

__attribute__((interrupt_save_fp("ABORT"))) void test_abort_interrupt() {
  // CHECK: define{{.*}} arm_aapcscc void @test_abort_interrupt() [[ABORT_ATTR:#[0-9]+]]
}


__attribute__((interrupt_save_fp("UNDEF"))) void test_undef_interrupt() {
  // CHECK: define{{.*}} arm_aapcscc void @test_undef_interrupt() [[UNDEF_ATTR:#[0-9]+]]
}


// CHECK: attributes [[GENERIC_ATTR]] = { {{.*}} {{"interrupt"[^=]}}{{.*}} "save-fp"
// CHECK: attributes [[IRQ_ATTR]] = { {{.*}} "interrupt"="IRQ" {{.*}} "save-fp"
// CHECK: attributes [[FIQ_ATTR]] = { {{.*}} "interrupt"="FIQ" {{.*}} "save-fp"
// CHECK: attributes [[SWI_ATTR]] = { {{.*}} "interrupt"="SWI" {{.*}} "save-fp"
// CHECK: attributes [[ABORT_ATTR]] = { {{.*}} "interrupt"="ABORT" {{.*}} "save-fp"
// CHECK: attributes [[UNDEF_ATTR]] = { {{.*}} "interrupt"="UNDEF" {{.*}} "save-fp"

// CHECK-APCS: attributes [[GENERIC_ATTR]] = { {{.*}} "interrupt" {{.*}} "save-fp"