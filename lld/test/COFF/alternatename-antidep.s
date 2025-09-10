// REQUIRES: x86

// Check that an anti-dependency alias can't be used as an alternate name target.
// RUN: llvm-mc -filetype=obj -triple=x86_64-windows %s -o %t.obj
// RUN: not lld-link -dll -noentry %t.obj -alternatename:sym=altsym 2>&1 | FileCheck %s
// CHECK: error: undefined symbol: sym

        .data
        .rva sym

        .weak_anti_dep altsym
        .set altsym,a

        .globl a
a:
        .word 1
