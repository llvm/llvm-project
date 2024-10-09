# REQUIRES: hexagon
# RUN: llvm-mc -filetype=obj -triple=hexagon-unknown-elf %s -o %t.o
# RUN: ld.lld %t.o -o %t
# RUN: llvm-objdump -d %t 2>&1 | FileCheck --check-prefix=CHECK-NONPIC %s
# RUN: llvm-mc -filetype=obj --position-independent \
# RUN:         -triple=hexagon-unknown-elf %s -o %t.o

# RUN: ld.lld --pie %t.o -o %t
# RUN: llvm-objdump -d %t 2>&1 | FileCheck --check-prefix=CHECK-PIC %s

    .globl main
    .type  main, @function
main:
    call myfn
    jumpr r31
    .size   main, .-main

    .org 0x800000

    .globl myfn
    .type  myfn, @function
myfn:
    jumpr r31
    .size  myfn, .-myfn

# CHECK:  Disassembly of section .text:

# CHECK-NONPIC:  000200b4 <__trampoline_for_myfn_from_.text.thunk>:
# CHECK-NONPIC:  { immext(#0x800000)
# CHECK-NONPIC:    jump 0x8200bc }
# CHECK-PIC:     00010150 <__trampoline_for_myfn_from_.text.thunk>:
# CHECK-PIC:     { immext(#0x800000)
# CHECK-PIC:       r14 = add(pc,##0x80000c) }
# CHECK-PIC:     { jumpr r14 }

# CHECK-NONPIC:  000200bc <main>:
# CHECK-NONPIC:    call 0x200b4
# CHECK-PIC:     0001015c <main>:
# CHECK-PIC:       call 0x10150
# CHECK:           jumpr r31

# CHECK-NONPIC:  008200bc <myfn>:
# CHECK-PIC:     0081015c <myfn>:
# CHECK:           jumpr r31
