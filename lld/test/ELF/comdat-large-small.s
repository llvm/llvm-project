# REQUIRES: x86
## Test that when the same comdat group exists in both small and large versions
## (with and without SHF_X86_64_LARGE), the linker prefers the small version.
## This is important for mixed small/medium code model linking.

# RUN: rm -rf %t && split-file %s %t && cd %t

# RUN: llvm-mc -filetype=obj -triple=x86_64 small.s -o small.o
# RUN: llvm-mc -filetype=obj -triple=x86_64 large.s -o large.o
# RUN: llvm-mc -filetype=obj -triple=x86_64 large2.s -o large2.o

## Test small first, then large - small should win
# RUN: ld.lld -e _start small.o large.o -o sl
# RUN: llvm-readelf -S -s sl | FileCheck %s --check-prefix=SMALL

## Test large first, then small - small should still win
# RUN: ld.lld -e _start large.o small.o -o ls
# RUN: llvm-readelf -S -s ls | FileCheck %s --check-prefix=SMALL

## When both are large, first one wins (no preference)
# RUN: ld.lld -e _start large.o large2.o -o ll
# RUN: llvm-readelf -S ll | FileCheck %s --check-prefix=LARGE

## Verify the symbol is in .data (not .ldata) and is defined
# SMALL:      .data{{.*}} PROGBITS
# SMALL-NOT:  .ldata
# SMALL:      OBJECT  GLOBAL DEFAULT [[#]] comdat_var

## When both are large, we get .ldata
# LARGE:      .ldata{{.*}} PROGBITS

## Test comdat group with multiple symbols (symbol name != signature).
## This tests that ALL symbols in the comdat are handled correctly,
## not just the one matching the signature.
# RUN: llvm-mc -filetype=obj -triple=x86_64 small_multi.s -o small_multi.o
# RUN: llvm-mc -filetype=obj -triple=x86_64 large_multi.s -o large_multi.o

## Large first, then small - small should win and both symbols should be defined
# RUN: ld.lld -e _start large_multi.o small_multi.o -o multi
# RUN: llvm-readelf -S -s multi | FileCheck %s --check-prefix=MULTI

# MULTI:      .data{{.*}} PROGBITS
# MULTI-NOT:  .ldata
## Both symbols from the comdat group should be properly defined
# MULTI-DAG:  OBJECT  GLOBAL DEFAULT [[#]] primary_var
# MULTI-DAG:  OBJECT  GLOBAL DEFAULT [[#]] secondary_var

#--- small.s
.globl _start
_start:
  # Access comdat_var with 32-bit PC-relative relocation (small code model)
  movl comdat_var(%rip), %eax
  ret

.section .data.comdat_var,"awG",@progbits,comdat_var,comdat
.globl comdat_var
.type comdat_var, @object
comdat_var:
  .long 42
  .size comdat_var, 4

#--- large.s
.globl use_large
use_large:
  # Access comdat_var with 64-bit absolute relocation (medium code model)
  movabsq $comdat_var, %rax
  movl (%rax), %eax
  ret

.section .ldata.comdat_var,"awlG",@progbits,comdat_var,comdat
.globl comdat_var
.type comdat_var, @object
comdat_var:
  .long 42
  .size comdat_var, 4

#--- large2.s
.globl _start
_start:
use_large2:
  movabsq $comdat_var, %rax
  movl (%rax), %eax
  ret

.section .ldata.comdat_var,"awlG",@progbits,comdat_var,comdat
.globl comdat_var
.type comdat_var, @object
comdat_var:
  .long 42
  .size comdat_var, 4

#--- small_multi.s
## Comdat group with signature "comdat_group" containing two symbols.
## The symbol names (primary_var, secondary_var) differ from the signature.
.globl _start
_start:
  movl primary_var(%rip), %eax
  addl secondary_var(%rip), %eax
  ret

.section .data.comdat_group,"awG",@progbits,comdat_group,comdat
.globl primary_var
.type primary_var, @object
primary_var:
  .long 1
  .size primary_var, 4

.globl secondary_var
.type secondary_var, @object
secondary_var:
  .long 2
  .size secondary_var, 4

#--- large_multi.s
## Large version of the same comdat group.
.globl use_large_multi
use_large_multi:
  movabsq $primary_var, %rax
  movl (%rax), %eax
  movabsq $secondary_var, %rcx
  addl (%rcx), %eax
  ret

.section .ldata.comdat_group,"awlG",@progbits,comdat_group,comdat
.globl primary_var
.type primary_var, @object
primary_var:
  .long 1
  .size primary_var, 4

.globl secondary_var
.type secondary_var, @object
secondary_var:
  .long 2
  .size secondary_var, 4
