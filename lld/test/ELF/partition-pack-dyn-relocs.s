// Test that Android and RELR packed relocation sections still emit correctly
// for the main partition; the shim has no input sections so its rela/relr
// sections are empty by construction.

// REQUIRES: x86

// RUN: llvm-mc %s -o %t.o -filetype=obj --triple=x86_64-unknown-linux
// RUN: ld.lld %t.o -o %t --shared --gc-sections --pack-dyn-relocs=android+relr

// RUN: llvm-objcopy --extract-main-partition %t %t0
// RUN: llvm-objcopy --extract-partition=part1 %t %t1

// RUN: llvm-readelf --all %t0 | FileCheck --check-prefix=MAIN %s
// RUN: llvm-readelf --all %t1 | FileCheck --check-prefix=SHIM %s

/// Main retains both packed sections referencing every relocatable global.
// MAIN: Section Headers:
// MAIN: .rela.dyn      ANDROID_RELA
// MAIN: .relr.dyn      RELR
// MAIN: .data          PROGBITS
// MAIN: Relocation section '.rela.dyn'
// MAIN: R_X86_64_64 {{.*}} p0 + 0
// MAIN: R_X86_64_64 {{.*}} p1 + 0
// MAIN: Relocation section '.relr.dyn'
// MAIN-DAG: p0
// MAIN-DAG: p1

/// The shim has no input sections. Enumerate every section and program
/// header so any spurious .rela.dyn / .relr.dyn / ANDROID_RELA would fail.
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

.section .llvm_sympart,"",@llvm_sympart
.asciz "part1"
.quad p1

.section .data.p0,"aw",@progbits
.align 8
.globl p0
p0:
.quad __ehdr_start
.quad p0

.section .data.p1,"aw",@progbits
.align 8
.globl p1
p1:
.quad __ehdr_start
.quad p1
