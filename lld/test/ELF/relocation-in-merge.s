// REQUIRES: x86
// RUN: llvm-mc %s -o %t.o -filetype=obj -triple=x86_64-pc-linux
// RUN: not ld.lld2 %t.o -o %t 2>&1 | FileCheck %s
// CHECK: Relocations pointing to SHF_MERGE are not supported

        .section	.foo,"aM",@progbits,4
        .long bar
