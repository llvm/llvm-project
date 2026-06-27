// Registers R0 - R3 have different names inside the LLVM MSP430 target code.
// Test that they are handled properly when used inside clobber lists.
// At the time of writing, llc silently ignores unknown register names.

// REQUIRES: msp430-registered-target
// RUN: %clang -target msp430 -c %s -mllvm -stop-after=finalize-isel -o- | FileCheck %s

void test_function(void) {
  asm volatile(""
               :
               :
               : "r0");
  asm volatile(""
               :
               :
               : "r1");
  asm volatile(""
               :
               :
               : "r2");
  asm volatile(""
               :
               :
               : "r3");
  asm volatile(""
               :
               :
               : "r4");
  asm volatile(""
               :
               :
               : "r5");
  asm volatile(""
               :
               :
               : "r6");
  asm volatile(""
               :
               :
               : "r7");
  asm volatile(""
               :
               :
               : "r8");
  asm volatile(""
               :
               :
               : "r9");
  asm volatile(""
               :
               :
               : "r10");
  asm volatile(""
               :
               :
               : "r11");
  asm volatile(""
               :
               :
               : "r12");
  asm volatile(""
               :
               :
               : "r13");
  asm volatile(""
               :
               :
               : "r14");
  asm volatile(""
               :
               :
               : "r15");
  // CHECK: call void asm sideeffect "", "~{pc}"()
  // CHECK: call void asm sideeffect "", "~{sp}"()
  // CHECK: call void asm sideeffect "", "~{sr}"()
  // CHECK: call void asm sideeffect "", "~{cg}"()
  // CHECK: call void asm sideeffect "", "~{r4}"()
  // CHECK: call void asm sideeffect "", "~{r5}"()
  // CHECK: call void asm sideeffect "", "~{r6}"()
  // CHECK: call void asm sideeffect "", "~{r7}"()
  // CHECK: call void asm sideeffect "", "~{r8}"()
  // CHECK: call void asm sideeffect "", "~{r9}"()
  // CHECK: call void asm sideeffect "", "~{r10}"()
  // CHECK: call void asm sideeffect "", "~{r11}"()
  // CHECK: call void asm sideeffect "", "~{r12}"()
  // CHECK: call void asm sideeffect "", "~{r13}"()
  // CHECK: call void asm sideeffect "", "~{r14}"()
  // CHECK: call void asm sideeffect "", "~{r15}"()
  // CHECK: INLINEASM &"", {{.*}} implicit-def $pc
  // CHECK: INLINEASM &"", {{.*}} implicit-def $sp
  // CHECK: INLINEASM &"", {{.*}} implicit-def $sr
  // CHECK: INLINEASM &"", {{.*}} implicit-def $cg
  // CHECK: INLINEASM &"", {{.*}} implicit-def $r4
  // CHECK: INLINEASM &"", {{.*}} implicit-def $r5
  // CHECK: INLINEASM &"", {{.*}} implicit-def $r6
  // CHECK: INLINEASM &"", {{.*}} implicit-def $r7
  // CHECK: INLINEASM &"", {{.*}} implicit-def $r8
  // CHECK: INLINEASM &"", {{.*}} implicit-def $r9
  // CHECK: INLINEASM &"", {{.*}} implicit-def $r10
  // CHECK: INLINEASM &"", {{.*}} implicit-def $r11
  // CHECK: INLINEASM &"", {{.*}} implicit-def $r12
  // CHECK: INLINEASM &"", {{.*}} implicit-def $r13
  // CHECK: INLINEASM &"", {{.*}} implicit-def $r14
  // CHECK: INLINEASM &"", {{.*}} implicit-def $r15
}
