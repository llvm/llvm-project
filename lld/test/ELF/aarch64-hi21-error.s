// RUN: llvm-mc -filetype=obj -triple=aarch64-pc-freebsd %S/Inputs/abs.s -o %tabs
// RUN: llvm-mc -filetype=obj -triple=aarch64-pc-freebsd %s -o %t
// RUN: not ld.lld2 %tabs -shared %t -o %t2 2>&1 | FileCheck %s
// REQUIRES: aarch64

adrp x0, big

#CHECK: R_AARCH64_ADR_PREL_PG_HI21 out of range
