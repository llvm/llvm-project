// REQUIRES: aarch64-registered-target

// RUN: %clang_cc1 -triple aarch64-linux-gnu -emit-llvm %s -o - | FileCheck --check-prefix=CHECK-IR %s
// RUN: %clang_cc1 -triple aarch64-linux-gnu -S -ffunction-sections %s -o - | FileCheck --check-prefix=CHECK-ASM %s
// RUN: %clang_cc1 -triple aarch64-linux-gnu -emit-llvm %s -o - | FileCheck --check-prefix=CHECK-EXPLICIT %s --implicit-check-not='@ctor_explicit_section(){{.*}}section_prefix'

// CHECK-IR: define{{.*}} void @plain_ctor(){{.*}} !section_prefix ![[CTOR:[0-9]+]]
// CHECK-IR: define{{.*}} void @plain_dtor(){{.*}} !section_prefix ![[DTOR:[0-9]+]]
// CHECK-IR: define{{.*}} void @ctor_prio(){{.*}} !section_prefix ![[CTOR]]
// CHECK-IR: define{{.*}} void @dtor_prio(){{.*}} !section_prefix ![[DTOR]]
// CHECK-IR: ![[CTOR]] = !{!"section_prefix", !"startup"}
// CHECK-IR: ![[DTOR]] = !{!"section_prefix", !"exit"}

// CHECK-ASM: .section .text.startup.plain_ctor,"ax",@progbits
// CHECK-ASM: .section .text.exit.plain_dtor,"ax",@progbits
// CHECK-ASM: .section .text.startup.ctor_prio,"ax",@progbits
// CHECK-ASM: .section .text.exit.dtor_prio,"ax",@progbits

// CHECK-EXPLICIT: define{{.*}} void @ctor_explicit_section(){{.*}} section ".my_section"

void __attribute__((constructor)) plain_ctor(void) {}

void __attribute__((destructor)) plain_dtor(void) {}

void __attribute__((constructor(101))) ctor_prio(void) {}

void __attribute__((destructor(202))) dtor_prio(void) {}

void __attribute__((constructor, section(".my_section"))) ctor_explicit_section(void) {}
