// RUN: %clang %cflags -o %t %s
// RUN: link_fdata --no-lbr %s %t %t.fdata
// RUN: llvm-bolt %t -o /dev/null --data=%t.fdata --dyno-stats | FileCheck %s

// CHECK: BOLT-INFO: program-wide dynostats after all optimizations before SCTC and FOP (no change):
// CHECK: 3000 : executed instructions
// CHECK: 1000 : executed load instructions
// CHECK: 1000 : executed store instructions

    .globl _start
_start:
# FDATA: 1 _start #_start# 1
    ld t0, (gp)
    sd t0, (gp)
    ret
    .size _start, .-_start
