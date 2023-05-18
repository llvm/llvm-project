// REQUIRES: arm-registered-target
// RUN: %clang_cc1 -triple armv6-unknown-unknown -emit-llvm -o - %s | FileCheck %s

void test0(void) {
	asm volatile("mov r0, r0" :: );
}
void test1(void) {
	asm volatile("mov r0, r0" :::
				 "cc", "memory" );
}
void test2(void) {
	asm volatile("mov r0, r0" :::
				 "r0", "r1", "r2", "r3");
	asm volatile("mov r0, r0" :::
				 "r4", "r5", "r6", "r8");
}
void test3(void) {
	asm volatile("mov r0, r0" :::
				 "a1", "a2", "a3", "a4");
	asm volatile("mov r0, r0" :::
				 "v1", "v2", "v3", "v5");
}


// {} should not be treated as asm variants.
void test4(float *a, float *b) {
  // CHECK: @test4
  // CHECK: call void asm sideeffect "vld1.32 {d8[],d9[]}, 
  __asm__ volatile (
                    "vld1.32 {d8[],d9[]}, [%1,:32] \n\t"
                    "vst1.32 {q4},        [%0,:128] \n\t"
                    :: "r"(a), "r"(b));
}

// {sp, lr, pc} are the canonical names for {r13, r14, r15}.
//
// CHECK: @test5
// CHECK: call void asm sideeffect "", "~{sp},~{lr},~{pc},~{sp},~{lr},~{pc}"()
void test5() {
  __asm__("" : : : "r13", "r14", "r15", "sp", "lr", "pc");
}

// CHECK: @test6
// CHECK: call void asm sideeffect "", "
// CHECK: ~{s0},~{s1},~{s2},~{s3},~{s4},~{s5},~{s6},~{s7},
// CHECK: ~{s8},~{s9},~{s10},~{s11},~{s12},~{s13},~{s14},~{s15},
// CHECK: ~{s16},~{s17},~{s18},~{s19},~{s20},~{s21},~{s22},~{s23},
// CHECK: ~{s24},~{s25},~{s26},~{s27},~{s28},~{s29},~{s30},~{s31}"()
void test6() {
  __asm__("" : : :
          "s0", "s1", "s2", "s3", "s4", "s5", "s6", "s7",
          "s8", "s9", "s10", "s11", "s12", "s13", "s14", "s15",
          "s16", "s17", "s18", "s19", "s20", "s21", "s22", "s23",
          "s24", "s25", "s26", "s27", "s28", "s29", "s30", "s31");
}
