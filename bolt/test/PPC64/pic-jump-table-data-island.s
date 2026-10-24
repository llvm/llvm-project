## Test that BOLT treats a GCC PIC switch table embedded in a function body as
## data instead of decoding it as instructions.
##
## On PPC64 ELFv2, GCC places a switch statement's jump table inside the
## function, directly after the indirect branch that reads it:
##
##   lwax  rDst, rBase, rIndex   # read a signed 32-bit table entry
##   add   rDst, rDst, rBase     # the entry is an offset from the table base
##   mtctr rDst
##   bctr                        # jump to it
##   .long ...                   # <-- table data starts here
##   <real code continues here>
##
## The table words have primary opcode 0, so they do not decode as
## instructions. BOLT used to lose the instruction boundaries there and then
## fail to attach the function's CFI directives:
##
##   CFI pointing to unknown instruction
##
## BinaryFunction::disassemble() now recognizes the sequence, finds where the
## table ends, and marks the range as data with code resuming after it.
# REQUIRES: system-linux
# RUN: llvm-mc -filetype=obj -triple powerpc64le-unknown-linux-gnu %s -o %t.o
# RUN: ld.lld %t.o -o %t.exe -e _start --emit-relocs
# RUN: llvm-bolt %t.exe -o %t.bolt 2>&1 | FileCheck %s
# RUN: %t.bolt

# CHECK: BOLT-INFO: Target architecture: powerpc64le
# CHECK: BOLT-INFO: enabling relocation mode
## The table must not be decoded, and its CFI must still attach.
# CHECK-NOT: unable to disassemble instruction at offset
# CHECK-NOT: CFI pointing to unknown instruction
        .text
        .abiversion 2
## foo returns 0 for argument 0 and dispatches through a PIC jump table
## otherwise. Only the argument-0 path runs in this test; the point of the
## test is what BOLT does when it disassembles the rest.
        .globl foo
        .type  foo, @function
foo:
        .localentry foo, 1
        .cfi_startproc
        stdu    1, -32(1)
        .cfi_def_cfa_offset 32
        cmpldi  3, 0
        beq     0, .Lret
## The switch dispatch: load the table base, read a signed 32-bit entry, add
## the base back, and jump through CTR.
        addis   4, 2, .Ltable@toc@ha
        addi    4, 4, .Ltable@toc@l
        sldi    5, 3, 2
        lwax    6, 4, 5
        add     6, 6, 4
        mtctr   6
        bctr
## The jump table, inside the function body, immediately after the bctr.
## Both entries are non-zero, 4-byte aligned and smaller than the function,
## so the scan accepts them. The following "li 3, 1" is not 4-byte aligned as
## a value, which is what ends the table.
.Ltable:
        .long   .Lcase1-.Ltable
        .long   .Lcase2-.Ltable
.Lcase1:
        li      3, 1
        b       .Lepilogue
.Lcase2:
        li      3, 2
        b       .Lepilogue
.Lret:
        li      3, 0
## A CFI directive after the table. This is the one that had no instruction to
## attach to while the table was being decoded as instructions.
.Lepilogue:
        addi    1, 1, 32
        .cfi_def_cfa_offset 0
        blr
        .cfi_endproc
        .size foo, .-foo
## Entry point - calls foo with 0, then exits with foo's return value.
        .globl _start
        .type  _start, @function
_start:
        .localentry _start, 1
        li      3, 0
        bl      foo
        nop                     # TOC restore slot (required by ELFv2 ABI)
        li      0, 1            # syscall: exit
        sc                      # exit with foo's return value in r3
        .size _start, .-_start
