// RUN: %clang_cc1 -triple x86_64-unknown-unknown -emit-llvm -O2 %s -o - | FileCheck %s

// CHECK-LABEL: @test_r15
// CHECK: call void asm sideeffect "", "{r15},~{dirflag},~{fpsr},~{flags}"(i32 undef)
void test_r15() {
    register int a asm ("r15");
    asm ("" :: "r" (a));
}

// CHECK-LABEL: @test_r16
// CHECK: call void asm sideeffect "", "{r16},~{dirflag},~{fpsr},~{flags}"(i32 undef)
void test_r16() {
    register int a asm ("r16");
    asm ("" :: "r" (a));
}

// CHECK-LABEL: @test_r17
// CHECK: call void asm sideeffect "", "{r17},~{dirflag},~{fpsr},~{flags}"(i32 undef)
void test_r17() {
    register int a asm ("r17");
    asm ("" :: "r" (a));
}

// CHECK-LABEL: @test_r18
// CHECK: call void asm sideeffect "", "{r18},~{dirflag},~{fpsr},~{flags}"(i32 undef)
void test_r18() {
    register int a asm ("r18");
    asm ("" :: "r" (a));
}

// CHECK-LABEL: @test_r19
// CHECK: call void asm sideeffect "", "{r19},~{dirflag},~{fpsr},~{flags}"(i32 undef)
void test_r19() {
    register int a asm ("r19");
    asm ("" :: "r" (a));
}

// CHECK-LABEL: @test_r20
// CHECK: call void asm sideeffect "", "{r20},~{dirflag},~{fpsr},~{flags}"(i32 undef)
void test_r20() {
    register int a asm ("r20");
    asm ("" :: "r" (a));
}

// CHECK-LABEL: @test_r21
// CHECK: call void asm sideeffect "", "{r21},~{dirflag},~{fpsr},~{flags}"(i32 undef)
void test_r21() {
    register int a asm ("r21");
    asm ("" :: "r" (a));
}

// CHECK-LABEL: @test_r22
// CHECK: call void asm sideeffect "", "{r22},~{dirflag},~{fpsr},~{flags}"(i32 undef)
void test_r22() {
    register int a asm ("r22");
    asm ("" :: "r" (a));
}

// CHECK-LABEL: @test_r23
// CHECK: call void asm sideeffect "", "{r23},~{dirflag},~{fpsr},~{flags}"(i32 undef)
void test_r23() {
    register int a asm ("r23");
    asm ("" :: "r" (a));
}

// CHECK-LABEL: @test_r24
// CHECK: call void asm sideeffect "", "{r24},~{dirflag},~{fpsr},~{flags}"(i32 undef)
void test_r24() {
    register int a asm ("r24");
    asm ("" :: "r" (a));
}

// CHECK-LABEL: @test_r25
// CHECK: call void asm sideeffect "", "{r25},~{dirflag},~{fpsr},~{flags}"(i32 undef)
void test_r25() {
    register int a asm ("r25");
    asm ("" :: "r" (a));
}

// CHECK-LABEL: @test_r26
// CHECK: call void asm sideeffect "", "{r26},~{dirflag},~{fpsr},~{flags}"(i32 undef)
void test_r26() {
    register int a asm ("r26");
    asm ("" :: "r" (a));
}

// CHECK-LABEL: @test_r27
// CHECK: call void asm sideeffect "", "{r27},~{dirflag},~{fpsr},~{flags}"(i32 undef)
void test_r27() {
    register int a asm ("r27");
    asm ("" :: "r" (a));
}

// CHECK-LABEL: @test_r28
// CHECK: call void asm sideeffect "", "{r28},~{dirflag},~{fpsr},~{flags}"(i32 undef)
void test_r28() {
    register int a asm ("r28");
    asm ("" :: "r" (a));
}

// CHECK-LABEL: @test_r29
// CHECK: call void asm sideeffect "", "{r29},~{dirflag},~{fpsr},~{flags}"(i32 undef)
void test_r29() {
    register int a asm ("r29");
    asm ("" :: "r" (a));
}

// CHECK-LABEL: @test_r30
// CHECK: call void asm sideeffect "", "{r30},~{dirflag},~{fpsr},~{flags}"(i32 undef)
void test_r30() {
    register int a asm ("r30");
    asm ("" :: "r" (a));
}

// CHECK-LABEL: @test_r31
// CHECK: call void asm sideeffect "", "{r31},~{dirflag},~{fpsr},~{flags}"(i32 undef)
void test_r31() {
    register int a asm ("r31");
    asm ("" :: "r" (a));
}

