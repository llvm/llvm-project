// RUN: %clang_cc1 -ffreestanding -triple i386-unknown-unknown -fasm-blocks -O0 -emit-llvm -S %s -o - | FileCheck %s
// RUN: %clang_cc1 -ffreestanding -triple x86_64-unknown-unknown -fasm-blocks -O0 -emit-llvm -S %s -o - | FileCheck %s
// REQUIRES: x86-registered-target

#include <immintrin.h>

void f(void) {
  __asm mov eax, ebx
  __asm mov ebx, ecx
  __asm__("movl %ecx, %edx");
  // CHECK: movl    %ebx, %eax
  // CHECK: movl    %ecx, %ebx
  // CHECK: movl    %ecx, %edx

  __asm mov eax, ebx
  __asm volatile ("movl %ecx, %edx");
  // CHECK: movl    %ebx, %eax
  // CHECK: movl    %ecx, %edx

  __asm volatile goto ("movl %ecx, %edx");
  // CHECK: movl    %ecx, %edx

  __asm mov eax, ebx
  __asm goto ("movl %ecx, %edx");
  // CHECK: movl    %ebx, %eax
  // CHECK: movl    %ecx, %edx
}
