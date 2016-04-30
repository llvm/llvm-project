// REQUIRES: x86
// RUN: llvm-mc -filetype=obj -triple=x86_64-pc-linux %s -o %t.o
// RUN: not ld.lld %t.o %t.o -o %t -shared 2>&1 | FileCheck %s

        .section        bar,"axG",@progbits,foo,comdat
        .cfi_startproc
        .cfi_personality 0x8c, foo
        .cfi_endproc

// CHECK: relocation R_X86_64_64 cannot be used when making a shared object; recompile with -fPIC.
