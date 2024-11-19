// RUN: %clang_cc1 -ffreestanding -triple i386 -fasm-blocks -O0 -S %s -o - | FileCheck %s
// RUN: %clang_cc1 -ffreestanding -triple x86_64 -fasm-blocks -O0 -S %s -o - | FileCheck %s
// REQUIRES: x86-registered-target

void f(void) {
  int src = 0;
  int dst;
  [[clang::asm_dialect("intel")]] __asm__ (
    ".intel_syntax noprefix\n\t"
    "# f1\n\t"
    "mov %1, %0\n\t"
    "add %0, 1\n\t"
    : "=r" (dst)
    : "r" (src)
  );
  // CHECK:      # f1
  // CHECK-NEXT: movl %eax, %eax
  // CHECK-NEXT: addl $1, %eax
  [[clang::asm_dialect("att")]] __asm__ (
    ".att_syntax prefix\n\t"
    "# f2\n\t"
    "movl %1, %0\n\t"
    "addl $1, %0\n\t"
    : "=r" (dst)
    : "r" (src)
  );
  // CHECK:      # f2
  // CHECK-NEXT: movl %eax, %eax
  // CHECK-NEXT: addl $1, %eax
}


#pragma clang attribute push ([[clang::asm_dialect("intel")]], apply_to = function)
void intel_fn(void) {
  int src = 0;
  int dst;
  __asm__ (
    ".intel_syntax noprefix\n\t"
    "# intel_fn\n\t"
    "mov %1, %0\n\t"
    "add %0, 1\n\t"
    : "=r" (dst)
    : "r" (src)
  );
  // CHECK:      # intel_fn
  // CHECK-NEXT: movl %eax, %eax
  // CHECK-NEXT: addl $1, %eax
}
#pragma clang attribute pop
#pragma clang attribute push ([[clang::asm_dialect("att")]], apply_to = function)
void att_fn(void) {
  int src = 0;
  int dst;
  __asm__ (
    ".att_syntax prefix\n\t"
    "# att_fn\n\t"
    "movl %1, %0\n\t"
    "addl $1, %0\n\t"
    : "=r" (dst)
    : "r" (src)
  );
  // CHECK:      # att_fn
  // CHECK-NEXT: movl %eax, %eax
  // CHECK-NEXT: addl $1, %eax
}
#pragma clang attribute pop
