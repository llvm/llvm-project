// REQUIRES: x86

// RUN: llvm-mc -filetype=obj -triple=x86_64-pc-linux %s -o %t
// RUN: not ld.lld %t -o %t2 2>&1 | FileCheck %s

        .section .eh_frame
        .byte 0

// CHECK: CIE/FDE too small
