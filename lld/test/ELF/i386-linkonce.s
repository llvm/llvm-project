// REQUIRES: x86
// RUN: llvm-mc -filetype=obj -triple=i386-linux-gnu %s -o %t.o
// RUN: llvm-mc -filetype=obj -triple=i386-linux-gnu %p/Inputs/i386-linkonce.s -o %t2.o
// RUN: llvm-ar rcs %t2.a %t2.o
// RUN: not ld.lld %t.o %t2.a -o /dev/null 2>&1 | FileCheck %s

// CHECK: error: relocation refers to a symbol in a discarded section: __i686.get_pc_thunk.bx

    .globl _start
_start:
    call _strchr1
