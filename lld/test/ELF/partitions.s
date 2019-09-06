// REQUIRES: aarch64, x86
// RUN: llvm-mc %s -o %t.o -filetype=obj --triple=x86_64-unknown-linux
// RUN: ld.lld %t.o -o %t --export-dynamic --gc-sections -z max-page-size=65536
// RUN: llvm-readelf -S -s %t | FileCheck %s

// RUN: llvm-mc %s -o %t.o -filetype=obj --triple=aarch64
// RUN: ld.lld %t.o -o %t --export-dynamic --gc-sections
// RUN: llvm-readelf -S -s %t | FileCheck %s

// This is basically lld/docs/partitions.dot in object file form.
// Test that the sections are correctly allocated to partitions.
// part1 and part2 should be aligned to a maximum page size boundary.

// CHECK: [[MAIN:[0-9]+]]] .text
// CHECK: part1 LLVM_PART_EHDR {{[0-9a-f]+}}0000 {{[0-9a-f]+}}0000
// CHECK: [[P1:[0-9]+]]] .text
// CHECK: part2 LLVM_PART_EHDR {{[0-9a-f]+}}0000 {{[0-9a-f]+}}0000
// CHECK: [[P2:[0-9]+]]] .text

// CHECK: Symbol table '.symtab'
// CHECK: [[MAIN]] f3
// CHECK:   [[P1]] f4
// CHECK: [[MAIN]] f5
// CHECK:   [[P2]] f6
// CHECK: [[MAIN]] _start
// CHECK:   [[P1]] f1
// CHECK:   [[P2]] f2

.section .llvm_sympart.f1,"",@llvm_sympart
.asciz "part1"
.quad f1

.section .llvm_sympart.f2,"",@llvm_sympart
.asciz "part2"
.quad f2

.section .text._start,"ax",@progbits
.globl _start
_start:
.quad f3

.section .text.f1,"ax",@progbits
.globl f1
f1:
.quad f3
.quad f4
.quad f5

.section .text.f2,"ax",@progbits
.globl f2
f2:
.quad f3
.quad f5
.quad f6

.section .text.f3,"ax",@progbits
f3:
ret

.section .text.f4,"ax",@progbits
f4:
ret

.section .text.f5,"ax",@progbits
f5:
ret

.section .text.f6,"ax",@progbits
f6:
ret
