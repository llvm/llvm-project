// REQUIRES: x86
// RUN: llvm-mc %s -o %t.o -filetype=obj -triple=x86_64-pc-linux
// RUN: not ld.lld %t.o -o %t 2>&1 | FileCheck %s
// CHECK: Writable SHF_MERGE sections are not supported

        .section	.foo,"awM",@progbits,4
