// REQUIRES: x86
// RUN: llvm-mc -filetype=obj -triple=x86_64 %s -o %t.o
// RUN: not ld.lld %t.o -o /dev/null 2>&1 | FileCheck %s --implicit-check-not=error:

// CHECK:      error: corrupted .eh_frame: CIE/FDE too small
// CHECK-NEXT: >>> defined in {{.*}}:(.eh_frame+0x0)

.section .eh_frame,"a",@unwind
.byte 0
