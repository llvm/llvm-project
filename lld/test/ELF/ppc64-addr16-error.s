// RUN: llvm-mc -filetype=obj -triple=powerpc64-unknown-linux %s -o %t
// RUN: not ld.lld2 -shared %t -o %t2 2>&1 | FileCheck %s
// REQUIRES: ppc

.short sym+65539

// CHECK: Relocation R_PPC64_ADDR16 overflow
