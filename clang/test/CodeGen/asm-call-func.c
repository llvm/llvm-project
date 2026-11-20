// RUN: %clang_cc1 -emit-llvm %s -o - -triple x86_64-unknown-linux-gnu| FileCheck %s
// RUN: %if cir-enabled %{ %clang_cc1 -fclangir -emit-llvm %s -o - -triple x86_64-unknown-linux-gnu| FileCheck %s %}

void callee(void);
void caller(void) {
  //CHECK: call void asm sideeffect "rcall $0", "n,~{dirflag},~{fpsr},~{flags}"(ptr @callee)
  asm("rcall %0" ::"n"(callee));
}
