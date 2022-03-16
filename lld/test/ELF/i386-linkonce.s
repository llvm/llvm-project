// REQUIRES: x86
// RUN: llvm-mc -filetype=obj -triple=i386-linux-gnu %s -o %t.o
// RUN: llvm-mc -filetype=obj -triple=i386-linux-gnu %p/Inputs/i386-linkonce.s -o %t2.o
// RUN: llvm-ar rcs %t2.a %t2.o

/// crti.o in i386 glibc<2.32 has .gnu.linkonce.t.__x86.get_pc_thunk.bx that is
/// not fully supported. Test that we don't report
/// "relocation refers to a symbol in a discarded section: __x86.get_pc_thunk.bx".
// RUN: ld.lld %t.o %t2.a %t2.o -o /dev/null

    .globl _start
_start:
    call _strchr1
