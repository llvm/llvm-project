// REQUIRES: aarch64
// RUN: rm -rf %t && mkdir %t && cd %t
// RUN: llvm-mc -filetype=obj -triple=aarch64 %s -o a.o
// RUN: not ld.lld a.o --defsym big=0x1000000000 2>&1 | FileCheck %s --implicit-check-not=error:

// CHECK: error: a.o:(.text+0x0): relocation R_AARCH64_ADR_PREL_PG_HI21 out of range: {{.*}} is not in [-4294967296, 4294967295]; references 'big'

.globl _start
_start:
adrp x0, big
