// RUN: %clang_cc1 -triple hexagon-unknown-elf -target-feature +hvx -target-feature +hvx-length128b -emit-llvm -o - %s | FileCheck %s

void test_r0(void) {
  // CHECK: define {{.*}}void @test_r0
  // CHECK: call void asm sideeffect "nop", "~{r0}"()
  asm("nop" ::: "r0");
}
void test_r1(void) {
  // CHECK: define {{.*}}void @test_r1
  // CHECK: call void asm sideeffect "nop", "~{r1}"()
  asm("nop" ::: "r1");
}
void test_r2(void) {
  // CHECK: define {{.*}}void @test_r2
  // CHECK: call void asm sideeffect "nop", "~{r2}"()
  asm("nop" ::: "r2");
}
void test_r3(void) {
  // CHECK: define {{.*}}void @test_r3
  // CHECK: call void asm sideeffect "nop", "~{r3}"()
  asm("nop" ::: "r3");
}
void test_r4(void) {
  // CHECK: define {{.*}}void @test_r4
  // CHECK: call void asm sideeffect "nop", "~{r4}"()
  asm("nop" ::: "r4");
}
void test_r5(void) {
  // CHECK: define {{.*}}void @test_r5
  // CHECK: call void asm sideeffect "nop", "~{r5}"()
  asm("nop" ::: "r5");
}
void test_r6(void) {
  // CHECK: define {{.*}}void @test_r6
  // CHECK: call void asm sideeffect "nop", "~{r6}"()
  asm("nop" ::: "r6");
}
void test_r7(void) {
  // CHECK: define {{.*}}void @test_r7
  // CHECK: call void asm sideeffect "nop", "~{r7}"()
  asm("nop" ::: "r7");
}
void test_r8(void) {
  // CHECK: define {{.*}}void @test_r8
  // CHECK: call void asm sideeffect "nop", "~{r8}"()
  asm("nop" ::: "r8");
}
void test_r9(void) {
  // CHECK: define {{.*}}void @test_r9
  // CHECK: call void asm sideeffect "nop", "~{r9}"()
  asm("nop" ::: "r9");
}
void test_r10(void) {
  // CHECK: define {{.*}}void @test_r10
  // CHECK: call void asm sideeffect "nop", "~{r10}"()
  asm("nop" ::: "r10");
}
void test_r11(void) {
  // CHECK: define {{.*}}void @test_r11
  // CHECK: call void asm sideeffect "nop", "~{r11}"()
  asm("nop" ::: "r11");
}
void test_r12(void) {
  // CHECK: define {{.*}}void @test_r12
  // CHECK: call void asm sideeffect "nop", "~{r12}"()
  asm("nop" ::: "r12");
}
void test_r13(void) {
  // CHECK: define {{.*}}void @test_r13
  // CHECK: call void asm sideeffect "nop", "~{r13}"()
  asm("nop" ::: "r13");
}
void test_r14(void) {
  // CHECK: define {{.*}}void @test_r14
  // CHECK: call void asm sideeffect "nop", "~{r14}"()
  asm("nop" ::: "r14");
}
void test_r15(void) {
  // CHECK: define {{.*}}void @test_r15
  // CHECK: call void asm sideeffect "nop", "~{r15}"()
  asm("nop" ::: "r15");
}
void test_r16(void) {
  // CHECK: define {{.*}}void @test_r16
  // CHECK: call void asm sideeffect "nop", "~{r16}"()
  asm("nop" ::: "r16");
}
void test_r17(void) {
  // CHECK: define {{.*}}void @test_r17
  // CHECK: call void asm sideeffect "nop", "~{r17}"()
  asm("nop" ::: "r17");
}
void test_r18(void) {
  // CHECK: define {{.*}}void @test_r18
  // CHECK: call void asm sideeffect "nop", "~{r18}"()
  asm("nop" ::: "r18");
}
void test_r19(void) {
  // CHECK: define {{.*}}void @test_r19
  // CHECK: call void asm sideeffect "nop", "~{r19}"()
  asm("nop" ::: "r19");
}
void test_r20(void) {
  // CHECK: define {{.*}}void @test_r20
  // CHECK: call void asm sideeffect "nop", "~{r20}"()
  asm("nop" ::: "r20");
}
void test_r21(void) {
  // CHECK: define {{.*}}void @test_r21
  // CHECK: call void asm sideeffect "nop", "~{r21}"()
  asm("nop" ::: "r21");
}
void test_r22(void) {
  // CHECK: define {{.*}}void @test_r22
  // CHECK: call void asm sideeffect "nop", "~{r22}"()
  asm("nop" ::: "r22");
}
void test_r23(void) {
  // CHECK: define {{.*}}void @test_r23
  // CHECK: call void asm sideeffect "nop", "~{r23}"()
  asm("nop" ::: "r23");
}
void test_r24(void) {
  // CHECK: define {{.*}}void @test_r24
  // CHECK: call void asm sideeffect "nop", "~{r24}"()
  asm("nop" ::: "r24");
}
void test_r25(void) {
  // CHECK: define {{.*}}void @test_r25
  // CHECK: call void asm sideeffect "nop", "~{r25}"()
  asm("nop" ::: "r25");
}
void test_r26(void) {
  // CHECK: define {{.*}}void @test_r26
  // CHECK: call void asm sideeffect "nop", "~{r26}"()
  asm("nop" ::: "r26");
}
void test_r27(void) {
  // CHECK: define {{.*}}void @test_r27
  // CHECK: call void asm sideeffect "nop", "~{r27}"()
  asm("nop" ::: "r27");
}
void test_r28(void) {
  // CHECK: define {{.*}}void @test_r28
  // CHECK: call void asm sideeffect "nop", "~{r28}"()
  asm("nop" ::: "r28");
}
void test_r29(void) {
  // CHECK: define {{.*}}void @test_r29
  // CHECK: call void asm sideeffect "nop", "~{r29}"()
  asm("nop" ::: "r29");
}
void test_r30(void) {
  // CHECK: define {{.*}}void @test_r30
  // CHECK: call void asm sideeffect "nop", "~{r30}"()
  asm("nop" ::: "r30");
}
void test_r31(void) {
  // CHECK: define {{.*}}void @test_r31
  // CHECK: call void asm sideeffect "nop", "~{r31}"()
  asm("nop" ::: "r31");
}
void test_r1_0(void) {
  // CHECK: define {{.*}}void @test_r1_0
  // CHECK: call void asm sideeffect "nop", "~{r1:0}"()
  asm("nop" ::: "r1:0");
}
void test_r3_2(void) {
  // CHECK: define {{.*}}void @test_r3_2
  // CHECK: call void asm sideeffect "nop", "~{r3:2}"()
  asm("nop" ::: "r3:2");
}
void test_r5_4(void) {
  // CHECK: define {{.*}}void @test_r5_4
  // CHECK: call void asm sideeffect "nop", "~{r5:4}"()
  asm("nop" ::: "r5:4");
}
void test_r7_6(void) {
  // CHECK: define {{.*}}void @test_r7_6
  // CHECK: call void asm sideeffect "nop", "~{r7:6}"()
  asm("nop" ::: "r7:6");
}
void test_r9_8(void) {
  // CHECK: define {{.*}}void @test_r9_8
  // CHECK: call void asm sideeffect "nop", "~{r9:8}"()
  asm("nop" ::: "r9:8");
}
void test_r11_10(void) {
  // CHECK: define {{.*}}void @test_r11_10
  // CHECK: call void asm sideeffect "nop", "~{r11:10}"()
  asm("nop" ::: "r11:10");
}
void test_r13_12(void) {
  // CHECK: define {{.*}}void @test_r13_12
  // CHECK: call void asm sideeffect "nop", "~{r13:12}"()
  asm("nop" ::: "r13:12");
}
void test_r15_14(void) {
  // CHECK: define {{.*}}void @test_r15_14
  // CHECK: call void asm sideeffect "nop", "~{r15:14}"()
  asm("nop" ::: "r15:14");
}
void test_r17_16(void) {
  // CHECK: define {{.*}}void @test_r17_16
  // CHECK: call void asm sideeffect "nop", "~{r17:16}"()
  asm("nop" ::: "r17:16");
}
void test_r19_18(void) {
  // CHECK: define {{.*}}void @test_r19_18
  // CHECK: call void asm sideeffect "nop", "~{r19:18}"()
  asm("nop" ::: "r19:18");
}
void test_r21_20(void) {
  // CHECK: define {{.*}}void @test_r21_20
  // CHECK: call void asm sideeffect "nop", "~{r21:20}"()
  asm("nop" ::: "r21:20");
}
void test_r23_22(void) {
  // CHECK: define {{.*}}void @test_r23_22
  // CHECK: call void asm sideeffect "nop", "~{r23:22}"()
  asm("nop" ::: "r23:22");
}
void test_r25_24(void) {
  // CHECK: define {{.*}}void @test_r25_24
  // CHECK: call void asm sideeffect "nop", "~{r25:24}"()
  asm("nop" ::: "r25:24");
}
void test_r27_26(void) {
  // CHECK: define {{.*}}void @test_r27_26
  // CHECK: call void asm sideeffect "nop", "~{r27:26}"()
  asm("nop" ::: "r27:26");
}
void test_r29_28(void) {
  // CHECK: define {{.*}}void @test_r29_28
  // CHECK: call void asm sideeffect "nop", "~{r29:28}"()
  asm("nop" ::: "r29:28");
}
void test_r31_30(void) {
  // CHECK: define {{.*}}void @test_r31_30
  // CHECK: call void asm sideeffect "nop", "~{r31:30}"()
  asm("nop" ::: "r31:30");
}
void test_p0(void) {
  // CHECK: define {{.*}}void @test_p0
  // CHECK: call void asm sideeffect "nop", "~{p0}"()
  asm("nop" ::: "p0");
}
void test_p1(void) {
  // CHECK: define {{.*}}void @test_p1
  // CHECK: call void asm sideeffect "nop", "~{p1}"()
  asm("nop" ::: "p1");
}
void test_p2(void) {
  // CHECK: define {{.*}}void @test_p2
  // CHECK: call void asm sideeffect "nop", "~{p2}"()
  asm("nop" ::: "p2");
}
void test_p3(void) {
  // CHECK: define {{.*}}void @test_p3
  // CHECK: call void asm sideeffect "nop", "~{p3}"()
  asm("nop" ::: "p3");
}
void test_c0(void) {
  // CHECK: define {{.*}}void @test_c0
  // CHECK: call void asm sideeffect "nop", "~{c0}"()
  asm("nop" ::: "c0");
}
void test_c1(void) {
  // CHECK: define {{.*}}void @test_c1
  // CHECK: call void asm sideeffect "nop", "~{c1}"()
  asm("nop" ::: "c1");
}
void test_c2(void) {
  // CHECK: define {{.*}}void @test_c2
  // CHECK: call void asm sideeffect "nop", "~{c2}"()
  asm("nop" ::: "c2");
}
void test_c3(void) {
  // CHECK: define {{.*}}void @test_c3
  // CHECK: call void asm sideeffect "nop", "~{c3}"()
  asm("nop" ::: "c3");
}
void test_c4(void) {
  // CHECK: define {{.*}}void @test_c4
  // CHECK: call void asm sideeffect "nop", "~{c4}"()
  asm("nop" ::: "c4");
}
void test_c5(void) {
  // CHECK: define {{.*}}void @test_c5
  // CHECK: call void asm sideeffect "nop", "~{c5}"()
  asm("nop" ::: "c5");
}
void test_c6(void) {
  // CHECK: define {{.*}}void @test_c6
  // CHECK: call void asm sideeffect "nop", "~{c6}"()
  asm("nop" ::: "c6");
}
void test_c7(void) {
  // CHECK: define {{.*}}void @test_c7
  // CHECK: call void asm sideeffect "nop", "~{c7}"()
  asm("nop" ::: "c7");
}
void test_c8(void) {
  // CHECK: define {{.*}}void @test_c8
  // CHECK: call void asm sideeffect "nop", "~{c8}"()
  asm("nop" ::: "c8");
}
void test_c9(void) {
  // CHECK: define {{.*}}void @test_c9
  // CHECK: call void asm sideeffect "nop", "~{c9}"()
  asm("nop" ::: "c9");
}
void test_c10(void) {
  // CHECK: define {{.*}}void @test_c10
  // CHECK: call void asm sideeffect "nop", "~{c10}"()
  asm("nop" ::: "c10");
}
void test_c11(void) {
  // CHECK: define {{.*}}void @test_c11
  // CHECK: call void asm sideeffect "nop", "~{c11}"()
  asm("nop" ::: "c11");
}
void test_c12(void) {
  // CHECK: define {{.*}}void @test_c12
  // CHECK: call void asm sideeffect "nop", "~{c12}"()
  asm("nop" ::: "c12");
}
void test_c13(void) {
  // CHECK: define {{.*}}void @test_c13
  // CHECK: call void asm sideeffect "nop", "~{c13}"()
  asm("nop" ::: "c13");
}
void test_c14(void) {
  // CHECK: define {{.*}}void @test_c14
  // CHECK: call void asm sideeffect "nop", "~{c14}"()
  asm("nop" ::: "c14");
}
void test_c15(void) {
  // CHECK: define {{.*}}void @test_c15
  // CHECK: call void asm sideeffect "nop", "~{c15}"()
  asm("nop" ::: "c15");
}
void test_c16(void) {
  // CHECK: define {{.*}}void @test_c16
  // CHECK: call void asm sideeffect "nop", "~{c16}"()
  asm("nop" ::: "c16");
}
void test_c17(void) {
  // CHECK: define {{.*}}void @test_c17
  // CHECK: call void asm sideeffect "nop", "~{c17}"()
  asm("nop" ::: "c17");
}
void test_c18(void) {
  // CHECK: define {{.*}}void @test_c18
  // CHECK: call void asm sideeffect "nop", "~{c18}"()
  asm("nop" ::: "c18");
}
void test_c19(void) {
  // CHECK: define {{.*}}void @test_c19
  // CHECK: call void asm sideeffect "nop", "~{c19}"()
  asm("nop" ::: "c19");
}
void test_c20(void) {
  // CHECK: define {{.*}}void @test_c20
  // CHECK: call void asm sideeffect "nop", "~{c20}"()
  asm("nop" ::: "c20");
}
void test_c21(void) {
  // CHECK: define {{.*}}void @test_c21
  // CHECK: call void asm sideeffect "nop", "~{c21}"()
  asm("nop" ::: "c21");
}
void test_c22(void) {
  // CHECK: define {{.*}}void @test_c22
  // CHECK: call void asm sideeffect "nop", "~{c22}"()
  asm("nop" ::: "c22");
}
void test_c23(void) {
  // CHECK: define {{.*}}void @test_c23
  // CHECK: call void asm sideeffect "nop", "~{c23}"()
  asm("nop" ::: "c23");
}
void test_c24(void) {
  // CHECK: define {{.*}}void @test_c24
  // CHECK: call void asm sideeffect "nop", "~{c24}"()
  asm("nop" ::: "c24");
}
void test_c25(void) {
  // CHECK: define {{.*}}void @test_c25
  // CHECK: call void asm sideeffect "nop", "~{c25}"()
  asm("nop" ::: "c25");
}
void test_c26(void) {
  // CHECK: define {{.*}}void @test_c26
  // CHECK: call void asm sideeffect "nop", "~{c26}"()
  asm("nop" ::: "c26");
}
void test_c27(void) {
  // CHECK: define {{.*}}void @test_c27
  // CHECK: call void asm sideeffect "nop", "~{c27}"()
  asm("nop" ::: "c27");
}
void test_c28(void) {
  // CHECK: define {{.*}}void @test_c28
  // CHECK: call void asm sideeffect "nop", "~{c28}"()
  asm("nop" ::: "c28");
}
void test_c29(void) {
  // CHECK: define {{.*}}void @test_c29
  // CHECK: call void asm sideeffect "nop", "~{c29}"()
  asm("nop" ::: "c29");
}
void test_c30(void) {
  // CHECK: define {{.*}}void @test_c30
  // CHECK: call void asm sideeffect "nop", "~{c30}"()
  asm("nop" ::: "c30");
}
void test_c31(void) {
  // CHECK: define {{.*}}void @test_c31
  // CHECK: call void asm sideeffect "nop", "~{c31}"()
  asm("nop" ::: "c31");
}
void test_c1_0(void) {
  // CHECK: define {{.*}}void @test_c1_0
  // CHECK: call void asm sideeffect "nop", "~{c1:0}"()
  asm("nop" ::: "c1:0");
}
void test_c3_2(void) {
  // CHECK: define {{.*}}void @test_c3_2
  // CHECK: call void asm sideeffect "nop", "~{c3:2}"()
  asm("nop" ::: "c3:2");
}
void test_c5_4(void) {
  // CHECK: define {{.*}}void @test_c5_4
  // CHECK: call void asm sideeffect "nop", "~{c5:4}"()
  asm("nop" ::: "c5:4");
}
void test_c7_6(void) {
  // CHECK: define {{.*}}void @test_c7_6
  // CHECK: call void asm sideeffect "nop", "~{c7:6}"()
  asm("nop" ::: "c7:6");
}
void test_c9_8(void) {
  // CHECK: define {{.*}}void @test_c9_8
  // CHECK: call void asm sideeffect "nop", "~{c9:8}"()
  asm("nop" ::: "c9:8");
}
void test_c11_10(void) {
  // CHECK: define {{.*}}void @test_c11_10
  // CHECK: call void asm sideeffect "nop", "~{c11:10}"()
  asm("nop" ::: "c11:10");
}
void test_c13_12(void) {
  // CHECK: define {{.*}}void @test_c13_12
  // CHECK: call void asm sideeffect "nop", "~{c13:12}"()
  asm("nop" ::: "c13:12");
}
void test_c15_14(void) {
  // CHECK: define {{.*}}void @test_c15_14
  // CHECK: call void asm sideeffect "nop", "~{c15:14}"()
  asm("nop" ::: "c15:14");
}
void test_c17_16(void) {
  // CHECK: define {{.*}}void @test_c17_16
  // CHECK: call void asm sideeffect "nop", "~{c17:16}"()
  asm("nop" ::: "c17:16");
}
void test_c19_18(void) {
  // CHECK: define {{.*}}void @test_c19_18
  // CHECK: call void asm sideeffect "nop", "~{c19:18}"()
  asm("nop" ::: "c19:18");
}
void test_c21_20(void) {
  // CHECK: define {{.*}}void @test_c21_20
  // CHECK: call void asm sideeffect "nop", "~{c21:20}"()
  asm("nop" ::: "c21:20");
}
void test_c23_22(void) {
  // CHECK: define {{.*}}void @test_c23_22
  // CHECK: call void asm sideeffect "nop", "~{c23:22}"()
  asm("nop" ::: "c23:22");
}
void test_c25_24(void) {
  // CHECK: define {{.*}}void @test_c25_24
  // CHECK: call void asm sideeffect "nop", "~{c25:24}"()
  asm("nop" ::: "c25:24");
}
void test_c27_26(void) {
  // CHECK: define {{.*}}void @test_c27_26
  // CHECK: call void asm sideeffect "nop", "~{c27:26}"()
  asm("nop" ::: "c27:26");
}
void test_c29_28(void) {
  // CHECK: define {{.*}}void @test_c29_28
  // CHECK: call void asm sideeffect "nop", "~{c29:28}"()
  asm("nop" ::: "c29:28");
}
void test_c31_30(void) {
  // CHECK: define {{.*}}void @test_c31_30
  // CHECK: call void asm sideeffect "nop", "~{c31:30}"()
  asm("nop" ::: "c31:30");
}
void test_sa0(void) {
  // CHECK: define {{.*}}void @test_sa0
  // CHECK: call void asm sideeffect "nop", "~{sa0}"()
  asm("nop" ::: "sa0");
}
void test_lc0(void) {
  // CHECK: define {{.*}}void @test_lc0
  // CHECK: call void asm sideeffect "nop", "~{lc0}"()
  asm("nop" ::: "lc0");
}
void test_sa1(void) {
  // CHECK: define {{.*}}void @test_sa1
  // CHECK: call void asm sideeffect "nop", "~{sa1}"()
  asm("nop" ::: "sa1");
}
void test_lc1(void) {
  // CHECK: define {{.*}}void @test_lc1
  // CHECK: call void asm sideeffect "nop", "~{lc1}"()
  asm("nop" ::: "lc1");
}
void test_p3_0(void) {
  // CHECK: define {{.*}}void @test_p3_0
  // CHECK: call void asm sideeffect "nop", "~{p3:0}"()
  asm("nop" ::: "p3:0");
}
void test_m0(void) {
  // CHECK: define {{.*}}void @test_m0
  // CHECK: call void asm sideeffect "nop", "~{m0}"()
  asm("nop" ::: "m0");
}
void test_m1(void) {
  // CHECK: define {{.*}}void @test_m1
  // CHECK: call void asm sideeffect "nop", "~{m1}"()
  asm("nop" ::: "m1");
}
void test_usr(void) {
  // CHECK: define {{.*}}void @test_usr
  // CHECK: call void asm sideeffect "nop", "~{usr}"()
  asm("nop" ::: "usr");
}
void test_pc(void) {
  // CHECK: define {{.*}}void @test_pc
  // CHECK: call void asm sideeffect "nop", "~{pc}"()
  asm("nop" ::: "pc");
}
void test_ugp(void) {
  // CHECK: define {{.*}}void @test_ugp
  // CHECK: call void asm sideeffect "nop", "~{ugp}"()
  asm("nop" ::: "ugp");
}
void test_gp(void) {
  // CHECK: define {{.*}}void @test_gp
  // CHECK: call void asm sideeffect "nop", "~{gp}"()
  asm("nop" ::: "gp");
}
void test_cs0(void) {
  // CHECK: define {{.*}}void @test_cs0
  // CHECK: call void asm sideeffect "nop", "~{cs0}"()
  asm("nop" ::: "cs0");
}
void test_cs1(void) {
  // CHECK: define {{.*}}void @test_cs1
  // CHECK: call void asm sideeffect "nop", "~{cs1}"()
  asm("nop" ::: "cs1");
}
void test_upcyclelo(void) {
  // CHECK: define {{.*}}void @test_upcyclelo
  // CHECK: call void asm sideeffect "nop", "~{upcyclelo}"()
  asm("nop" ::: "upcyclelo");
}
void test_upcyclehi(void) {
  // CHECK: define {{.*}}void @test_upcyclehi
  // CHECK: call void asm sideeffect "nop", "~{upcyclehi}"()
  asm("nop" ::: "upcyclehi");
}
void test_framelimit(void) {
  // CHECK: define {{.*}}void @test_framelimit
  // CHECK: call void asm sideeffect "nop", "~{framelimit}"()
  asm("nop" ::: "framelimit");
}
void test_framekey(void) {
  // CHECK: define {{.*}}void @test_framekey
  // CHECK: call void asm sideeffect "nop", "~{framekey}"()
  asm("nop" ::: "framekey");
}
void test_pktcountlo(void) {
  // CHECK: define {{.*}}void @test_pktcountlo
  // CHECK: call void asm sideeffect "nop", "~{pktcountlo}"()
  asm("nop" ::: "pktcountlo");
}
void test_pktcounthi(void) {
  // CHECK: define {{.*}}void @test_pktcounthi
  // CHECK: call void asm sideeffect "nop", "~{pktcounthi}"()
  asm("nop" ::: "pktcounthi");
}
void test_utimerlo(void) {
  // CHECK: define {{.*}}void @test_utimerlo
  // CHECK: call void asm sideeffect "nop", "~{utimerlo}"()
  asm("nop" ::: "utimerlo");
}
void test_utimerhi(void) {
  // CHECK: define {{.*}}void @test_utimerhi
  // CHECK: call void asm sideeffect "nop", "~{utimerhi}"()
  asm("nop" ::: "utimerhi");
}
void test_upcycle(void) {
  // CHECK: define {{.*}}void @test_upcycle
  // CHECK: call void asm sideeffect "nop", "~{upcycle}"()
  asm("nop" ::: "upcycle");
}
void test_pktcount(void) {
  // CHECK: define {{.*}}void @test_pktcount
  // CHECK: call void asm sideeffect "nop", "~{pktcount}"()
  asm("nop" ::: "pktcount");
}
void test_utimer(void) {
  // CHECK: define {{.*}}void @test_utimer
  // CHECK: call void asm sideeffect "nop", "~{utimer}"()
  asm("nop" ::: "utimer");
}
void test_v0(void) {
  // CHECK: define {{.*}}void @test_v0
  // CHECK: call void asm sideeffect "nop", "~{v0}"()
  asm("nop" ::: "v0");
}
void test_v1(void) {
  // CHECK: define {{.*}}void @test_v1
  // CHECK: call void asm sideeffect "nop", "~{v1}"()
  asm("nop" ::: "v1");
}
void test_v2(void) {
  // CHECK: define {{.*}}void @test_v2
  // CHECK: call void asm sideeffect "nop", "~{v2}"()
  asm("nop" ::: "v2");
}
void test_v3(void) {
  // CHECK: define {{.*}}void @test_v3
  // CHECK: call void asm sideeffect "nop", "~{v3}"()
  asm("nop" ::: "v3");
}
void test_v4(void) {
  // CHECK: define {{.*}}void @test_v4
  // CHECK: call void asm sideeffect "nop", "~{v4}"()
  asm("nop" ::: "v4");
}
void test_v5(void) {
  // CHECK: define {{.*}}void @test_v5
  // CHECK: call void asm sideeffect "nop", "~{v5}"()
  asm("nop" ::: "v5");
}
void test_v6(void) {
  // CHECK: define {{.*}}void @test_v6
  // CHECK: call void asm sideeffect "nop", "~{v6}"()
  asm("nop" ::: "v6");
}
void test_v7(void) {
  // CHECK: define {{.*}}void @test_v7
  // CHECK: call void asm sideeffect "nop", "~{v7}"()
  asm("nop" ::: "v7");
}
void test_v8(void) {
  // CHECK: define {{.*}}void @test_v8
  // CHECK: call void asm sideeffect "nop", "~{v8}"()
  asm("nop" ::: "v8");
}
void test_v9(void) {
  // CHECK: define {{.*}}void @test_v9
  // CHECK: call void asm sideeffect "nop", "~{v9}"()
  asm("nop" ::: "v9");
}
void test_v10(void) {
  // CHECK: define {{.*}}void @test_v10
  // CHECK: call void asm sideeffect "nop", "~{v10}"()
  asm("nop" ::: "v10");
}
void test_v11(void) {
  // CHECK: define {{.*}}void @test_v11
  // CHECK: call void asm sideeffect "nop", "~{v11}"()
  asm("nop" ::: "v11");
}
void test_v12(void) {
  // CHECK: define {{.*}}void @test_v12
  // CHECK: call void asm sideeffect "nop", "~{v12}"()
  asm("nop" ::: "v12");
}
void test_v13(void) {
  // CHECK: define {{.*}}void @test_v13
  // CHECK: call void asm sideeffect "nop", "~{v13}"()
  asm("nop" ::: "v13");
}
void test_v14(void) {
  // CHECK: define {{.*}}void @test_v14
  // CHECK: call void asm sideeffect "nop", "~{v14}"()
  asm("nop" ::: "v14");
}
void test_v15(void) {
  // CHECK: define {{.*}}void @test_v15
  // CHECK: call void asm sideeffect "nop", "~{v15}"()
  asm("nop" ::: "v15");
}
void test_v16(void) {
  // CHECK: define {{.*}}void @test_v16
  // CHECK: call void asm sideeffect "nop", "~{v16}"()
  asm("nop" ::: "v16");
}
void test_v17(void) {
  // CHECK: define {{.*}}void @test_v17
  // CHECK: call void asm sideeffect "nop", "~{v17}"()
  asm("nop" ::: "v17");
}
void test_v18(void) {
  // CHECK: define {{.*}}void @test_v18
  // CHECK: call void asm sideeffect "nop", "~{v18}"()
  asm("nop" ::: "v18");
}
void test_v19(void) {
  // CHECK: define {{.*}}void @test_v19
  // CHECK: call void asm sideeffect "nop", "~{v19}"()
  asm("nop" ::: "v19");
}
void test_v20(void) {
  // CHECK: define {{.*}}void @test_v20
  // CHECK: call void asm sideeffect "nop", "~{v20}"()
  asm("nop" ::: "v20");
}
void test_v21(void) {
  // CHECK: define {{.*}}void @test_v21
  // CHECK: call void asm sideeffect "nop", "~{v21}"()
  asm("nop" ::: "v21");
}
void test_v22(void) {
  // CHECK: define {{.*}}void @test_v22
  // CHECK: call void asm sideeffect "nop", "~{v22}"()
  asm("nop" ::: "v22");
}
void test_v23(void) {
  // CHECK: define {{.*}}void @test_v23
  // CHECK: call void asm sideeffect "nop", "~{v23}"()
  asm("nop" ::: "v23");
}
void test_v24(void) {
  // CHECK: define {{.*}}void @test_v24
  // CHECK: call void asm sideeffect "nop", "~{v24}"()
  asm("nop" ::: "v24");
}
void test_v25(void) {
  // CHECK: define {{.*}}void @test_v25
  // CHECK: call void asm sideeffect "nop", "~{v25}"()
  asm("nop" ::: "v25");
}
void test_v26(void) {
  // CHECK: define {{.*}}void @test_v26
  // CHECK: call void asm sideeffect "nop", "~{v26}"()
  asm("nop" ::: "v26");
}
void test_v27(void) {
  // CHECK: define {{.*}}void @test_v27
  // CHECK: call void asm sideeffect "nop", "~{v27}"()
  asm("nop" ::: "v27");
}
void test_v28(void) {
  // CHECK: define {{.*}}void @test_v28
  // CHECK: call void asm sideeffect "nop", "~{v28}"()
  asm("nop" ::: "v28");
}
void test_v29(void) {
  // CHECK: define {{.*}}void @test_v29
  // CHECK: call void asm sideeffect "nop", "~{v29}"()
  asm("nop" ::: "v29");
}
void test_v30(void) {
  // CHECK: define {{.*}}void @test_v30
  // CHECK: call void asm sideeffect "nop", "~{v30}"()
  asm("nop" ::: "v30");
}
void test_v31(void) {
  // CHECK: define {{.*}}void @test_v31
  // CHECK: call void asm sideeffect "nop", "~{v31}"()
  asm("nop" ::: "v31");
}
void test_v1_0(void) {
  // CHECK: define {{.*}}void @test_v1_0
  // CHECK: call void asm sideeffect "nop", "~{v1:0}"()
  asm("nop" ::: "v1:0");
}
void test_v3_2(void) {
  // CHECK: define {{.*}}void @test_v3_2
  // CHECK: call void asm sideeffect "nop", "~{v3:2}"()
  asm("nop" ::: "v3:2");
}
void test_v5_4(void) {
  // CHECK: define {{.*}}void @test_v5_4
  // CHECK: call void asm sideeffect "nop", "~{v5:4}"()
  asm("nop" ::: "v5:4");
}
void test_v7_6(void) {
  // CHECK: define {{.*}}void @test_v7_6
  // CHECK: call void asm sideeffect "nop", "~{v7:6}"()
  asm("nop" ::: "v7:6");
}
void test_v9_8(void) {
  // CHECK: define {{.*}}void @test_v9_8
  // CHECK: call void asm sideeffect "nop", "~{v9:8}"()
  asm("nop" ::: "v9:8");
}
void test_v11_10(void) {
  // CHECK: define {{.*}}void @test_v11_10
  // CHECK: call void asm sideeffect "nop", "~{v11:10}"()
  asm("nop" ::: "v11:10");
}
void test_v13_12(void) {
  // CHECK: define {{.*}}void @test_v13_12
  // CHECK: call void asm sideeffect "nop", "~{v13:12}"()
  asm("nop" ::: "v13:12");
}
void test_v15_14(void) {
  // CHECK: define {{.*}}void @test_v15_14
  // CHECK: call void asm sideeffect "nop", "~{v15:14}"()
  asm("nop" ::: "v15:14");
}
void test_v17_16(void) {
  // CHECK: define {{.*}}void @test_v17_16
  // CHECK: call void asm sideeffect "nop", "~{v17:16}"()
  asm("nop" ::: "v17:16");
}
void test_v19_18(void) {
  // CHECK: define {{.*}}void @test_v19_18
  // CHECK: call void asm sideeffect "nop", "~{v19:18}"()
  asm("nop" ::: "v19:18");
}
void test_v21_20(void) {
  // CHECK: define {{.*}}void @test_v21_20
  // CHECK: call void asm sideeffect "nop", "~{v21:20}"()
  asm("nop" ::: "v21:20");
}
void test_v23_22(void) {
  // CHECK: define {{.*}}void @test_v23_22
  // CHECK: call void asm sideeffect "nop", "~{v23:22}"()
  asm("nop" ::: "v23:22");
}
void test_v25_24(void) {
  // CHECK: define {{.*}}void @test_v25_24
  // CHECK: call void asm sideeffect "nop", "~{v25:24}"()
  asm("nop" ::: "v25:24");
}
void test_v27_26(void) {
  // CHECK: define {{.*}}void @test_v27_26
  // CHECK: call void asm sideeffect "nop", "~{v27:26}"()
  asm("nop" ::: "v27:26");
}
void test_v29_28(void) {
  // CHECK: define {{.*}}void @test_v29_28
  // CHECK: call void asm sideeffect "nop", "~{v29:28}"()
  asm("nop" ::: "v29:28");
}
void test_v31_30(void) {
  // CHECK: define {{.*}}void @test_v31_30
  // CHECK: call void asm sideeffect "nop", "~{v31:30}"()
  asm("nop" ::: "v31:30");
}
void test_v3_0(void) {
  // CHECK: define {{.*}}void @test_v3_0
  // CHECK: call void asm sideeffect "nop", "~{v3:0}"()
  asm("nop" ::: "v3:0");
}
void test_v7_4(void) {
  // CHECK: define {{.*}}void @test_v7_4
  // CHECK: call void asm sideeffect "nop", "~{v7:4}"()
  asm("nop" ::: "v7:4");
}
void test_v11_8(void) {
  // CHECK: define {{.*}}void @test_v11_8
  // CHECK: call void asm sideeffect "nop", "~{v11:8}"()
  asm("nop" ::: "v11:8");
}
void test_v15_12(void) {
  // CHECK: define {{.*}}void @test_v15_12
  // CHECK: call void asm sideeffect "nop", "~{v15:12}"()
  asm("nop" ::: "v15:12");
}
void test_v19_16(void) {
  // CHECK: define {{.*}}void @test_v19_16
  // CHECK: call void asm sideeffect "nop", "~{v19:16}"()
  asm("nop" ::: "v19:16");
}
void test_v23_20(void) {
  // CHECK: define {{.*}}void @test_v23_20
  // CHECK: call void asm sideeffect "nop", "~{v23:20}"()
  asm("nop" ::: "v23:20");
}
void test_v27_24(void) {
  // CHECK: define {{.*}}void @test_v27_24
  // CHECK: call void asm sideeffect "nop", "~{v27:24}"()
  asm("nop" ::: "v27:24");
}
void test_v31_28(void) {
  // CHECK: define {{.*}}void @test_v31_28
  // CHECK: call void asm sideeffect "nop", "~{v31:28}"()
  asm("nop" ::: "v31:28");
}
void test_q0(void) {
  // CHECK: define {{.*}}void @test_q0
  // CHECK: call void asm sideeffect "nop", "~{q0}"()
  asm("nop" ::: "q0");
}
void test_q1(void) {
  // CHECK: define {{.*}}void @test_q1
  // CHECK: call void asm sideeffect "nop", "~{q1}"()
  asm("nop" ::: "q1");
}
void test_q2(void) {
  // CHECK: define {{.*}}void @test_q2
  // CHECK: call void asm sideeffect "nop", "~{q2}"()
  asm("nop" ::: "q2");
}
void test_q3(void) {
  // CHECK: define {{.*}}void @test_q3
  // CHECK: call void asm sideeffect "nop", "~{q3}"()
  asm("nop" ::: "q3");
}
