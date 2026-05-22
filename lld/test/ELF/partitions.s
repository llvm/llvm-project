// REQUIRES: aarch64, x86
/// Link against a DSO to ensure that sections are not discarded by --gc-sections.
// RUN: llvm-mc %S/Inputs/shared.s -o %ts.o -filetype=obj --triple=x86_64
// RUN: ld.lld -shared -soname=ts %ts.o -o %ts.so
// RUN: llvm-mc %s -o %t.o -filetype=obj --triple=x86_64-unknown-linux
// RUN: ld.lld %t.o %ts.so -o %t --export-dynamic --gc-sections -z max-page-size=65536
// RUN: llvm-readelf -S -s %t | FileCheck %s

/// End-to-end round-trip: llvm-objcopy --extract-partition produces an
/// empty-but-valid shim .so; --extract-main-partition produces a main image
/// that carries every export and no SHT_LLVM_PART_* sections.
// RUN: llvm-objcopy --extract-main-partition %t %t.main
// RUN: llvm-objcopy --extract-partition=part1 %t %t.part1
// RUN: llvm-readelf -S -s %t.main | FileCheck --check-prefix=EXMAIN --implicit-check-not=LLVM_PART_EHDR --implicit-check-not=LLVM_PART_PHDR %s
// RUN: llvm-readelf -h -S -l -d -r -s %t.part1 | FileCheck --check-prefix=SHIM %s

// RUN: llvm-mc %S/Inputs/shared.s -o %ts.o -filetype=obj --triple=aarch64
// RUN: ld.lld -shared -soname=ts %ts.o -o %ts.so
// RUN: llvm-mc %s -o %t.o -filetype=obj --triple=aarch64 --crel
// RUN: ld.lld %t.o %ts.so -o %t --export-dynamic --gc-sections
// RUN: llvm-readelf -S -s %t | FileCheck %s

/// Every input section and every symbol now stays in the main partition.
/// Named SHT_LLVM_SYMPART inputs still emit empty-but-valid PART_EHDR shim
/// shells so `llvm-objcopy --extract-partition` keeps working.

// CHECK:      [[MAIN:[0-9]+]]] .text
// CHECK:      part1 LLVM_PART_EHDR
// CHECK:      part2 LLVM_PART_EHDR

// CHECK:      Symbol table '.symtab'
// CHECK-DAG:  [[MAIN]] f1
// CHECK-DAG:  [[MAIN]] f2
// CHECK-DAG:  [[MAIN]] f3
// CHECK-DAG:  [[MAIN]] f4
// CHECK-DAG:  [[MAIN]] f5
// CHECK-DAG:  [[MAIN]] f6
// CHECK-DAG:  [[MAIN]] _start

/// Extracted main carries every export and no PART_* sections.
// EXMAIN:      Symbol table '.dynsym'
// EXMAIN-DAG:  f1
// EXMAIN-DAG:  f2
// EXMAIN-DAG:  _start

/// Extracted shim is a minimal-but-parseable `.so`: ELF header, a PT_PHDR
/// and at least one PT_LOAD. No `.dynsym`, `.dynamic`, or notes — shim
/// partitions carry no per-partition content. Enumerate sections and
/// program headers with -NEXT so any spurious entry would fail the check.
// SHIM:      Type: DYN
// SHIM:      Section Headers:
// SHIM-NEXT: [Nr] Name              Type
// SHIM-NEXT: [ 0]                   NULL
// SHIM-NEXT: [ 1] .comment          PROGBITS
// SHIM-NEXT: [ 2] .shstrtab         STRTAB
// SHIM:      Program Headers:
// SHIM-NEXT: Type           Offset
// SHIM-NEXT: PHDR
// SHIM-NEXT: LOAD
// SHIM-NEXT: GNU_STACK
// SHIM:      There are no relocations in this file.

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
