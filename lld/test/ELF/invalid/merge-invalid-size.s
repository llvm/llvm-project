// REQUIRES: x86
// RUN: llvm-mc %s -o %t.o -filetype=obj -triple=x86_64-pc-linux
// RUN: not ld.lld %t.o -o /dev/null --noinhibit-exec 2>&1 | FileCheck %s
// CHECK: merge-invalid-size.s.tmp.o:(.foo): SHF_MERGE section size (2) must be a multiple of sh_entsize (4)

.section .foo,"aM",@progbits,4
.short 42
