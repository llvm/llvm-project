// RUN: llvm-mc %s -o %t.o -filetype=obj -triple=x86_64-pc-linux
// RUN: llvm-mc %p/Inputs/copy-rel-pie.s -o %t2.o -filetype=obj -triple=x86_64-pc-linux
// RUN: ld.lld %t2.o -o %t2.so -shared
// RUN: not ld.lld %t.o %t2.so -o %t.exe -pie 2>&1 | FileCheck %s

// CHECK: relocation R_X86_64_64 cannot be used when making a shared object; recompile with -fPIC.
// CHECK: relocation R_X86_64_64 cannot be used when making a shared object; recompile with -fPIC.

.global _start
_start:
        .quad bar
        .quad foo
