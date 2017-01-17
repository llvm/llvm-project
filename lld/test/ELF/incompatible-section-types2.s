// RUN: llvm-mc -filetype=obj -triple=x86_64-pc-linux %s -o %t.o
// RUN: not ld.lld %t.o -o %t 2>&1 | FileCheck %s

// CHECK: error: Section has different type from others with the same name <internal>:(.shstrtab)

.section .shstrtab,""
.short 20
