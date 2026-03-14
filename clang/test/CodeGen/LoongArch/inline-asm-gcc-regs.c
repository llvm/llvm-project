// RUN: %clang_cc1 -triple loongarch32 -emit-llvm -O2 %s -o - | FileCheck %s
// RUN: %clang_cc1 -triple loongarch64 -emit-llvm -O2 %s -o - | FileCheck %s

/// Check GCC register names and alias can be used in register variable definition.

// CHECK-LABEL: @test_r0
// CHECK-COUNT-2: call void asm sideeffect "", "{$r0}"(i32 undef)
// CHECK-NOT: call void asm sideeffect "", "{$r0}"(i32 undef)
void test_r0() {
    register int a asm ("$r0");
    register int b asm ("r0");
    asm ("" :: "r" (a));
    asm ("" :: "r" (b));
}

// CHECK-LABEL: @test_r12
// CHECK-COUNT-2: call void asm sideeffect "", "{$r12}"(i32 undef)
// CHECK-NOT: call void asm sideeffect "", "{$r12}"(i32 undef)
void test_r12() {
    register int a asm ("$r12");
    register int b asm ("r12");
    asm ("" :: "r" (a));
    asm ("" :: "r" (b));
}

// CHECK-LABEL: @test_r31
// CHECK-COUNT-2: call void asm sideeffect "", "{$r31}"(i32 undef)
// CHECK-NOT: call void asm sideeffect "", "{$r31}"(i32 undef)
void test_r31() {
    register int a asm ("$r31");
    register int b asm ("r31");
    asm ("" :: "r" (a));
    asm ("" :: "r" (b));
}

// CHECK-LABEL: @test_zero
// CHECK-COUNT-2: call void asm sideeffect "", "{$r0}"(i32 undef)
// CHECK-NOT: call void asm sideeffect "", "{$r0}"(i32 undef)
void test_zero() {
    register int a asm ("$zero");
    register int b asm ("zero");
    asm ("" :: "r" (a));
    asm ("" :: "r" (b));
}

// CHECK-LABEL: @test_a0
// CHECK-COUNT-2: call void asm sideeffect "", "{$r4}"(i32 undef)
// CHECK-NOT: call void asm sideeffect "", "{$r4}"(i32 undef)
void test_a0() {
    register int a asm ("$a0");
    register int b asm ("a0");
    asm ("" :: "r" (a));
    asm ("" :: "r" (b));
}

// CHECK-LABEL: @test_t1
// CHECK-COUNT-2: call void asm sideeffect "", "{$r13}"(i32 undef)
// CHECK-NOT: call void asm sideeffect "", "{$r13}"(i32 undef)
void test_t1() {
    register int a asm ("$t1");
    register int b asm ("t1");
    asm ("" :: "r" (a));
    asm ("" :: "r" (b));
}

// CHECK-LABEL: @test_fp
// CHECK-COUNT-2: call void asm sideeffect "", "{$r22}"(i32 undef)
// CHECK-NOT: call void asm sideeffect "", "{$r22}"(i32 undef)
void test_fp() {
    register int a asm ("$fp");
    register int b asm ("fp");
    asm ("" :: "r" (a));
    asm ("" :: "r" (b));
}

// CHECK-LABEL: @test_s2
// CHECK-COUNT-2: call void asm sideeffect "", "{$r25}"(i32 undef)
// CHECK-NOT: call void asm sideeffect "", "{$r25}"(i32 undef)
void test_s2() {
    register int a asm ("$s2");
    register int b asm ("s2");
    asm ("" :: "r" (a));
    asm ("" :: "r" (b));
}

// CHECK-LABEL: @test_f0
// CHECK-COUNT-2: call void asm sideeffect "", "{$f0}"(float undef)
// CHECK-NOT: call void asm sideeffect "", "{$f0}"(float undef)
void test_f0() {
    register float a asm ("$f0");
    register float b asm ("f0");
    asm ("" :: "f" (a));
    asm ("" :: "f" (b));
}

// CHECK-LABEL: @test_f14
// CHECK-COUNT-2: call void asm sideeffect "", "{$f14}"(float undef)
// CHECK-NOT: call void asm sideeffect "", "{$f14}"(float undef)
void test_f14() {
    register float a asm ("$f14");
    register float b asm ("f14");
    asm ("" :: "f" (a));
    asm ("" :: "f" (b));
}

// CHECK-LABEL: @test_f31
// CHECK-COUNT-2: call void asm sideeffect "", "{$f31}"(float undef)
// CHECK-NOT: call void asm sideeffect "", "{$f31}"(float undef)
void test_f31() {
    register float a asm ("$f31");
    register float b asm ("f31");
    asm ("" :: "f" (a));
    asm ("" :: "f" (b));
}

// CHECK-LABEL: @test_fa0
// CHECK-COUNT-2: call void asm sideeffect "", "{$f0}"(float undef)
// CHECK-NOT: call void asm sideeffect "", "{$f0}"(float undef)
void test_fa0() {
    register float a asm ("$fa0");
    register float b asm ("fa0");
    asm ("" :: "f" (a));
    asm ("" :: "f" (b));
}

// CHECK-LABEL: @test_ft1
// CHECK-COUNT-2: call void asm sideeffect "", "{$f9}"(float undef)
// CHECK-NOT: call void asm sideeffect "", "{$f9}"(float undef)
void test_ft1() {
    register float a asm ("$ft1");
    register float b asm ("ft1");
    asm ("" :: "f" (a));
    asm ("" :: "f" (b));
}

// CHECK-LABEL: @test_fs2
// CHECK-COUNT-2: call void asm sideeffect "", "{$f26}"(float undef)
// CHECK-NOT: call void asm sideeffect "", "{$f26}"(float undef)
void test_fs2() {
    register float a asm ("$fs2");
    register float b asm ("fs2");
    asm ("" :: "f" (a));
    asm ("" :: "f" (b));
}

// CHECK-LABEL: @test_fcc
// CHECK-COUNT-2: call void asm sideeffect "", "~{$fcc0}"()
// CHECK-NOT: call void asm sideeffect "", "~{$fcc0}"()
// CHECK-COUNT-2: call void asm sideeffect "", "~{$fcc7}"()
// CHECK-NOT: call void asm sideeffect "", "~{$fcc7}"()
void test_fcc() {
    asm ("" ::: "$fcc0");
    asm ("" ::: "fcc0");
    asm ("" ::: "$fcc7");
    asm ("" ::: "fcc7");
}
