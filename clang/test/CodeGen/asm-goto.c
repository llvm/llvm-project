// REQUIRES: x86-registered-target
// RUN: %clang_cc1 -triple x86_64-pc-linux-gnu -O0 -emit-llvm %s -o - | FileCheck %s
// RUN: %clang_cc1 -triple i386-pc-linux-gnu -O0 -emit-llvm %s -o - | FileCheck %s

int test1(int cond) {
  // CHECK-LABEL: define{{.*}} i32 @test1(
  // CHECK: callbr void asm sideeffect
  // CHECK: to label %asm.fallthrough [label %label_true, label %loop]
  // CHECK-LABEL: asm.fallthrough:
  asm volatile goto("testl %0, %0; jne %l1;" :: "r"(cond)::label_true, loop);
  asm volatile goto("testl %0, %0; jne %l2;" :: "r"(cond)::label_true, loop);
  // CHECK: callbr void asm sideeffect
  // CHECK: to label %asm.fallthrough1 [label %label_true, label %loop]
  // CHECK-LABEL: asm.fallthrough1:
  return 0;
loop:
  return 0;
label_true:
  return 1;
}

int test2(int cond) {
  // CHECK-LABEL: define{{.*}} i32 @test2(
  // CHECK: callbr i32 asm sideeffect
  // CHECK: to label %asm.fallthrough [label %label_true, label %loop]
  // CHECK-LABEL: asm.fallthrough:
  asm volatile goto("testl %0, %0; jne %l2;" : "=r"(cond) : "r"(cond) :: label_true, loop);
  asm volatile goto("testl %0, %0; jne %l3;" : "=r"(cond) : "r"(cond) :: label_true, loop);
  // CHECK: callbr i32 asm sideeffect
  // CHECK: to label %asm.fallthrough1 [label %label_true, label %loop]
  // CHECK-LABEL: asm.fallthrough1:
  return 0;
loop:
  return 0;
label_true:
  return 1;
}

int test3(int out1, int out2) {
  // CHECK-LABEL: define{{.*}} i32 @test3(
  // CHECK: callbr { i32, i32 } asm sideeffect
  // CHECK: to label %asm.fallthrough [label %label_true, label %loop]
  // CHECK-LABEL: asm.fallthrough:
  asm volatile goto("testl %0, %0; jne %l3;" : "=r"(out1), "=r"(out2) : "r"(out1) :: label_true, loop);
  asm volatile goto("testl %0, %0; jne %l4;" : "=r"(out1), "=r"(out2) : "r"(out1) :: label_true, loop);
  // CHECK: callbr { i32, i32 } asm sideeffect
  // CHECK: to label %asm.fallthrough2 [label %label_true, label %loop]
  // CHECK-LABEL: asm.fallthrough2:
  return 0;
loop:
  return 0;
label_true:
  return 1;
}

int test4(int out1, int out2) {
  // CHECK-LABEL: define{{.*}} i32 @test4(
  // CHECK: callbr { i32, i32 } asm sideeffect "jne ${5:l}", "={si},={di},r,0,1,!i,!i
  // CHECK: to label %asm.fallthrough [label %label_true, label %loop]
  // CHECK-LABEL: asm.fallthrough:
  if (out1 < out2)
    asm volatile goto("jne %l5" : "+S"(out1), "+D"(out2) : "r"(out1) :: label_true, loop);
  else
    asm volatile goto("jne %l7" : "+S"(out1), "+D"(out2) : "r"(out1), "r"(out2) :: label_true, loop);
  // CHECK: callbr { i32, i32 } asm sideeffect "jne ${7:l}", "={si},={di},r,r,0,1,!i,!i
  // CHECK: to label %asm.fallthrough2 [label %label_true, label %loop]
  // CHECK-LABEL: asm.fallthrough2:
  return out1 + out2;
loop:
  return -1;
label_true:
  return -2;
}

int test5(int addr, int size, int limit) {
  // CHECK-LABEL: define{{.*}} i32 @test5(
  // CHECK: callbr i32 asm "add $1,$0 ; jc ${4:l} ; cmp $2,$0 ; ja ${4:l} ; ", "=r,imr,imr,0,!i
  // CHECK: to label %asm.fallthrough [label %t_err]
  // CHECK-LABEL: asm.fallthrough:
  asm goto(
      "add %1,%0 ; "
      "jc %l[t_err] ; "
      "cmp %2,%0 ; "
      "ja %l[t_err] ; "
      : "+r" (addr)
      : "g" (size), "g" (limit)
      : : t_err);
  return 0;
t_err:
  return 1;
}

int test6(int out1) {
  // CHECK-LABEL: define{{.*}} i32 @test6(
  // CHECK: callbr i32 asm sideeffect "testl $0, $0; testl $1, $1; jne ${3:l}", "={si},r,0,!i,!i,{{.*}}
  // CHECK: to label %asm.fallthrough [label %label_true, label %landing]
  // CHECK-LABEL: asm.fallthrough:
  // CHECK-LABEL: landing:
  int out2 = 42;
  asm volatile goto("testl %0, %0; testl %1, %1; jne %l3" : "+S"(out2) : "r"(out1) :: label_true, landing);
landing:
  return out1 + out2;
label_true:
  return -2;
}

// test7 - For the output templates in the asm string (%0, %l2), GCC places
// hidden inputs tied to outputs ("+r" constraint) BEFORE labels. Test that foo
// is $2 (or rather ${2:l} because of the l output template) in the emitted asm
// string, not $1.
void *test7(void) {
  // CHECK-LABEL: define{{.*}} ptr @test7(
  // CHECK: %1 = callbr ptr asm "# $0\0A\09# ${2:l}", "=r,0,!i,~{dirflag},~{fpsr},~{flags}"(ptr %0)
  // CHECK-NEXT: to label %asm.fallthrough [label %foo]
  void *p = &&foo;
  asm goto ("# %0\n\t# %l2":"+r"(p):::foo);
foo:
  return p;
}

// test8 - the same as test7, but this time we use symbolic names rather than
// numbered outputs.
void *test8(void) {
  // CHECK-LABEL: define{{.*}} ptr @test8(
  // CHECK: %1 = callbr ptr asm "# $0\0A\09# ${2:l}", "=r,0,!i,~{dirflag},~{fpsr},~{flags}"(ptr %0)
  // CHECK-NEXT: to label %asm.fallthrough [label %foo]
  void *p = &&foo;
  asm goto ("# %0\n\t# %l[foo]":"+r"(p):::foo);
foo:
  return p;
}
