// REQUIRES: x86

// RUN: llvm-mc -filetype=obj -triple=x86_64-pc-linux %s -o %t
// RUN: not ld.lld %t -o %t2 2>&1 | FileCheck %s

 .section .eh_frame
 .long 0xFFFFFFFF
 .quad 1
 nop

// CHECK-NOT: Truncated CIE/FDE length
// CHECK-NOT: CIE/FIE size is too large
// CHECK-NOT: CIE/FIE ends past the end of the section
