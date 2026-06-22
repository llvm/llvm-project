// REQUIRES: x86
/// The symbol partition feature has been removed. lld no longer treats
/// SHT_LLVM_SYMPART specially: the marker sections are retained as ordinary
/// sections, no SHT_LLVM_PART_EHDR/PHDR output is produced, and every symbol
/// stays in the single output.
/// Link against a DSO to ensure that sections are not discarded by --gc-sections.
// RUN: llvm-mc %S/Inputs/shared.s -o %ts.o -filetype=obj --triple=x86_64
// RUN: ld.lld -shared -soname=ts %ts.o -o %ts.so
// RUN: llvm-mc %s -o %t.o -filetype=obj --triple=x86_64-unknown-linux
// RUN: ld.lld %t.o %ts.so -o %t --export-dynamic --gc-sections
// RUN: llvm-readelf -S -s %t | FileCheck %s \
// RUN:   --implicit-check-not=LLVM_PART_EHDR --implicit-check-not=LLVM_PART_PHDR

/// The SHT_LLVM_SYMPART markers pass through as ordinary sections.
// CHECK:      .llvm_sympart.f1 LLVM_SYMPART
// CHECK:      .llvm_sympart.f2 LLVM_SYMPART

// CHECK:      Symbol table '.symtab'
// CHECK-DAG:  _start
// CHECK-DAG:  f1
// CHECK-DAG:  f2
// CHECK-DAG:  f3
// CHECK-DAG:  f4
// CHECK-DAG:  f5
// CHECK-DAG:  f6

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
