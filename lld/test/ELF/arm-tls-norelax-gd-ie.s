// REQUIRES: arm
// RUN: llvm-mc -filetype=obj -triple=armv7a-none-linux-gnueabi %p/Inputs/arm-tls-get-addr.s -o %t1.o
// RUN: ld.lld %t1.o --shared -soname=t1.so -o %t1.so
// RUN: llvm-mc %s -o %t.o -filetype=obj -triple=armv7a-linux-gnueabi
// RUN: ld.lld %t1.so %t.o -o %t
// RUN: llvm-readobj -S --dyn-relocations %t | FileCheck %s

/// This tls global-dynamic sequence is with respect to a preemptible symbol but
/// is in an application so a relaxation to Initial Exec would normally be
/// possible. This would result in an assertion failure on ARM as the
/// relaxation functions can't be implemented on ARM. Check that the sequence
/// is handled as global dynamic

 .text
 .syntax unified
 .globl  func
 .p2align        2
 .type   func,%function
func:
.L0:
 .globl __tls_get_addr
 bl __tls_get_addr
 bx lr
 .p2align 2
 .Lt0: .word   y(TLSGD) + (. - .L0 - 8)

// CHECK: Dynamic Relocations {
// CHECK-NEXT:   0x30290 R_ARM_TLS_DTPMOD32 y
// CHECK-NEXT:   0x30294 R_ARM_TLS_DTPOFF32 y
// CHECK-NEXT:   0x402A4 R_ARM_JUMP_SLOT __tls_get_addr
