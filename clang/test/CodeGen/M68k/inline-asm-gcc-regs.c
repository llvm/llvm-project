// RUN: %clang_cc1 -triple m68k -emit-llvm -O2 %s -o - | FileCheck %s

/// Check GCC register names and alias can be used in register variable definition.

// CHECK-LABEL: @test_d0
// CHECK: call void asm sideeffect "", "{d0}"(i32 undef)
void test_d0() {
    register int a asm ("d0");
    asm ("" :: "r" (a));
}

// CHECK-LABEL: @test_d1
// CHECK: call void asm sideeffect "", "{d1}"(i32 undef)
void test_d1() {
    register int a asm ("d1");
    asm ("" :: "r" (a));
}

// CHECK-LABEL: @test_d2
// CHECK: call void asm sideeffect "", "{d2}"(i32 undef)
void test_d2() {
    register int a asm ("d2");
    asm ("" :: "r" (a));
}

// CHECK-LABEL: @test_d3
// CHECK: call void asm sideeffect "", "{d3}"(i32 undef)
void test_d3() {
    register int a asm ("d3");
    asm ("" :: "r" (a));
}

// CHECK-LABEL: @test_d4
// CHECK: call void asm sideeffect "", "{d4}"(i32 undef)
void test_d4() {
    register int a asm ("d4");
    asm ("" :: "r" (a));
}

// CHECK-LABEL: @test_d5
// CHECK: call void asm sideeffect "", "{d5}"(i32 undef)
void test_d5() {
    register int a asm ("d5");
    asm ("" :: "r" (a));
}

// CHECK-LABEL: @test_d6
// CHECK: call void asm sideeffect "", "{d6}"(i32 undef)
void test_d6() {
    register int a asm ("d6");
    asm ("" :: "r" (a));
}

// CHECK-LABEL: @test_d7
// CHECK: call void asm sideeffect "", "{d7}"(i32 undef)
void test_d7() {
    register int a asm ("d7");
    asm ("" :: "r" (a));
}

// CHECK-LABEL: @test_a0
// CHECK: call void asm sideeffect "", "{a0}"(i32 undef)
void test_a0() {
    register int a asm ("a0");
    asm ("" :: "r" (a));
}

// CHECK-LABEL: @test_a1
// CHECK: call void asm sideeffect "", "{a1}"(i32 undef)
void test_a1() {
    register int a asm ("a1");
    asm ("" :: "r" (a));
}

// CHECK-LABEL: @test_a2
// CHECK: call void asm sideeffect "", "{a2}"(i32 undef)
void test_a2() {
    register int a asm ("a2");
    asm ("" :: "r" (a));
}

// CHECK-LABEL: @test_a3
// CHECK: call void asm sideeffect "", "{a3}"(i32 undef)
void test_a3() {
    register int a asm ("a3");
    asm ("" :: "r" (a));
}

// CHECK-LABEL: @test_a4
// CHECK: call void asm sideeffect "", "{a4}"(i32 undef)
void test_a4() {
    register int a asm ("a4");
    asm ("" :: "r" (a));
}

// CHECK-LABEL: @test_a5
// CHECK: call void asm sideeffect "", "{a5}"(i32 undef)
void test_a5() {
    register int a asm ("a5");
    register int b asm ("bp");
    asm ("" :: "r" (a));
    asm ("" :: "r" (b));
}

// CHECK-LABEL: @test_a6
// CHECK: call void asm sideeffect "", "{a6}"(i32 undef)
void test_a6() {
    register int a asm ("a6");
    register int b asm ("fp");
    asm ("" :: "r" (a));
    asm ("" :: "r" (b));
}

// CHECK-LABEL: @test_sp
// CHECK: call void asm sideeffect "", "{sp}"(i32 undef)
void test_sp() {
    register int a asm ("sp");
    register int b asm ("usp");
    register int c asm ("ssp");
    register int d asm ("isp");
    register int e asm ("a7");
    asm ("" :: "r" (a));
    asm ("" :: "r" (b));
    asm ("" :: "r" (c));
    asm ("" :: "r" (d));
    asm ("" :: "r" (e));
}

// CHECK-LABEL: @test_pc
// CHECK: call void asm sideeffect "", "{pc}"(i32 undef)
void test_pc() {
    register int a asm ("pc");
    asm ("" :: "r" (a));
}
