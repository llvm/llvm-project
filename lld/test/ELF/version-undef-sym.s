// REQUIRES: x86
// RUN: llvm-mc -filetype=obj -triple=x86_64-pc-linux %s -o %t.o
// RUN: llvm-readobj --dyn-symbols %p/Inputs/version-undef-sym.so | FileCheck %s


// Show that the input .so has undefined symbols before bar. That is what would
// get our version parsing out of sync.

// CHECK: Section: Undefined
// CHECK: Section: Undefined
// CHECK: Section: Undefined
// CHECK: Section: Undefined
// CHECK: Section: Undefined
// CHECK: Name: bar

// But now we can successfully find bar.
// RUN: ld.lld %t.o %p/Inputs/version-undef-sym.so -o %t.exe

        .global _start
_start:
        call bar@plt
