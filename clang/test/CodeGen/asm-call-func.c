// RUN: %clang_cc1 -emit-llvm %s -o - -triple x86_64-unknown-linux-gnu| FileCheck %s

void callee(void);
void caller() {
  //CHECK: call void asm sideeffect "rcall $0", "n,~{dirflag},~{fpsr},~{flags}"(void ()* @callee)
  asm("rcall %0" ::"n"(callee));
}
