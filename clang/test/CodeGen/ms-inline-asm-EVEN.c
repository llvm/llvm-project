// REQUIRES: x86-registered-target
// RUN: %clang_cc1 %s -triple i386-apple-darwin10 -fasm-blocks -emit-llvm -o - | FileCheck %s

// CHECK: .byte 64
// CHECK: .byte 64
// CHECK: .byte 64
// CHECK:  .even
void t1(void) {
  __asm {
    .byte 64
    .byte 64
    .byte 64
    EVEN
    mov eax, ebx
  }
}
