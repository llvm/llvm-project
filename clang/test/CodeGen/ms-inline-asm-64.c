// REQUIRES: x86-64-registered-target
// RUN: %clang_cc1 %s -triple x86_64-apple-darwin10 -O0 -fms-extensions -fenable-experimental-ms-inline-asm -w -emit-llvm -o - | FileCheck %s

void t1() {
  int var = 10;
  __asm mov rax, offset var ; rax = address of myvar
// CHECK: t1
// CHECK: call void asm sideeffect inteldialect "mov rax, $0", "r,~{rax},~{dirflag},~{fpsr},~{flags}"(i32* %{{.*}}) nounwind
}

void t2() {
  int var = 10;
  __asm mov [eax], offset var
// CHECK: t2
// CHECK: call void asm sideeffect inteldialect "mov [eax], $0", "r,~{dirflag},~{fpsr},~{flags}"(i32* %{{.*}}) nounwind
}
