// REQUIRES: aarch64
// RUN: rm -rf %t && mkdir %t && cd %t
// RUN: llvm-mc -filetype=obj -triple=aarch64 %s -o a.o
// RUN: not ld.lld a.o --defsym big=0x1000000000 2>&1 | FileCheck %s --implicit-check-not=error:

// CHECK: error: a.o:(.text+0x0): relocation R_AARCH64_ADR_PREL_LO21 out of range: {{.*}} is not in [-1048576, 1048575]; references 'big'

.globl _start
_start:
adr x0, big
