// REQUIRES: x86

// Check that a weak alias can be used as an alternate name target.
// RUN: llvm-mc -filetype=obj -triple=x86_64-windows %s -o %t.obj
// RUN: lld-link -dll -noentry %t.obj -alternatename:sym=altsym

        .data
        .rva sym

        .weak altsym
        .set altsym,a

        .globl a
a:
        .word 1
