// REQUIRES: x86-registered-target
// RUN: %clang_cc1 %s -triple i386-apple-darwin10 -fasm-blocks -emit-llvm -o - | FileCheck %s

void t() {
  int eax;
  int Ecx;
  __asm mov eax, ebx
  // CHECK: mov $0, ebx
  __asm add ecx, Ecx
  // CHECK: add ecx, $1
}

