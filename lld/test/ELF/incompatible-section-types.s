// RUN: llvm-mc -filetype=obj -triple=x86_64-pc-linux %s -o %t.o
// RUN: not ld.lld -shared %t.o -o %t 2>&1 | FileCheck %s

// CHECK: error: Section has different type from others with the same name {{.*}}incompatible-section-types.s.tmp.o:(.foo)

.section .foo, "aw", @progbits, unique, 1
.quad 0

.section .foo, "aw", @init_array, unique, 2
.quad 0
