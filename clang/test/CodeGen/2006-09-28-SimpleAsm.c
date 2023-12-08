// RUN: %clang_cc1 %s -emit-llvm -o - | FileCheck %s
// PR924

void bar(void) {
  // Extended asm
  // CHECK: call void asm sideeffect "ext: xorl %eax, eax; movl eax, fs; movl eax, gs  %blah
  asm volatile ("ext: xorl %%eax, eax; movl eax, fs; movl eax, gs  %%blah %= %\
% " : : "r"(1));
  // CHECK: call void asm sideeffect "nonext: xorl %eax, %eax; movl %eax, %fs; movl %eax, %gs  %%blah %= %%
  // Non-extended asm.
  asm volatile ("nonext: xorl %eax, %eax; movl %eax, %fs; movl %eax, %gs  %%blah %= %% ");
}
