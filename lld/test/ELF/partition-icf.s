// REQUIRES: x86
/// Link against a DSO to ensure that sections are not discarded by --gc-sections.
// RUN: llvm-mc -filetype=obj -triple=x86_64 %S/Inputs/shared.s -o %ts.o
// RUN: ld.lld -shared -soname=ts %ts.o -o %ts.so
// RUN: llvm-mc %s -o %t.o -filetype=obj --triple=x86_64-unknown-linux
// RUN: ld.lld %t.o %ts.so -o %t --export-dynamic --gc-sections --icf=all
// RUN: llvm-readelf -S -s %t | FileCheck %s

/// Symbols stay in the main partition; ICF folds the identical bodies of
/// (f1, f2) and (g1, g2) into a single section even though they were tagged
/// with different SHT_LLVM_SYMPART entries.

// CHECK:      [[MAIN:[0-9]+]]] .text
// CHECK:      part1 LLVM_PART_EHDR
// CHECK:      part2 LLVM_PART_EHDR

// CHECK:      Symbol table '.symtab'
// CHECK-DAG:  [[MAIN]] f1
// CHECK-DAG:  [[MAIN]] f2
// CHECK-DAG:  [[MAIN]] g1
// CHECK-DAG:  [[MAIN]] g2

.section .llvm_sympart.f1,"",@llvm_sympart
.asciz "part1"
.quad f1

.section .llvm_sympart.f2,"",@llvm_sympart
.asciz "part2"
.quad f2

.section .llvm_sympart.g1,"",@llvm_sympart
.asciz "part1"
.quad g1

.section .llvm_sympart.g2,"",@llvm_sympart
.asciz "part2"
.quad g2

.section .text.f1,"ax",@progbits
.globl f1
f1:
.byte 1

.section .text.f2,"ax",@progbits
.globl f2
f2:
.byte 2

.section .text.g1,"ax",@progbits
.globl g1
g1:
.byte 3

.section .text.g2,"ax",@progbits
.globl g2
g2:
.byte 3
