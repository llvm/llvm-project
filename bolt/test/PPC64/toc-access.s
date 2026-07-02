## Test that BOLT correctly handles TOC-relative data access relocations.
## R_PPC64_TOC16_HA and R_PPC64_TOC16_LO are generated when accessing
## global data via the TOC (Table of Contents) pointer in r2.
## These always appear in pairs: addis loads the high part and
## addi/ld loads the low part of the TOC-relative offset.
# REQUIRES: system-linux
# RUN: llvm-mc -filetype=obj -triple powerpc64le-unknown-linux-gnu %s -o %t.o
# RUN: ld.lld %t.o -o %t.exe -e _start --emit-relocs
# RUN: llvm-bolt %t.exe -o %t.bolt -lite 2>&1 | FileCheck %s
# RUN: %t.bolt
# CHECK: BOLT-INFO: Target architecture: powerpc64le
# CHECK: BOLT-INFO: enabling relocation mode
        .text
        .abiversion 2
        .globl _start
        .type  _start, @function
_start:
        .localentry _start, 1
        addis   3, 2, .LC0@toc@ha    # R_PPC64_TOC16_HA
        addi    3, 3, .LC0@toc@l     # R_PPC64_TOC16_LO
        li      0, 1                  # syscall: exit
        li      3, 0                  # exit code 0
        sc
        .size _start, .-_start
        .section .toc,"aw"
.LC0:
        .quad   0
