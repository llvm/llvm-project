# REQUIRES: x86
# Tests to verify that
#    - the folded CUEs covered up to the last entries, even if they were removed/folded.
#    - only entries that *should* be in eh_frame are actually included. (**)
# (**) This is where LD64 does differently.

# RUN: rm -rf %t; split-file %s %t
# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-darwin %t/fold-tail.s -o %t/fold-tail.o
# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-darwin %t/cues-range-gap.s -o %t/cues-range-gap.o

# RUN: %lld -o %t/fold-tail.out %t/fold-tail.o
# RUN: llvm-objdump --macho --syms --unwind-info %t/fold-tail.out | FileCheck %s 

# RUN: %lld -o %t/cues-range-gap.out %t/cues-range-gap.o

# CHECK-LABEL: SYMBOL TABLE:
# CHECK:       [[#%x,A_ADDR:]]  l     F __TEXT,__text _a
# CHECK:       [[#%x,B_ADDR:]]  l     F __TEXT,__text _b
# CHECK:       [[#%x,C_ADDR:]]  l     F __TEXT,__text _c
# CHECK:       [[#%x,D_ADDR:]]  l     F __TEXT,__text _d
# CHECK:       [[#%x,MAIN_ADDR:]]  g     F __TEXT,__text _main

## Check that [1] offset starts at c's address + 3 (its length).
# CHECK-LABEL: Contents of __unwind_info section:
# CHECK:  Top level indices: (count = 2)
# CHECK-NEXT: [0]: function offset=[[#%#.7x,MAIN_ADDR]]
# CHECK-NEXT: [1]: function offset=[[#%#.7x,C_ADDR + 3]]

#--- fold-tail.s
        .text
        .globl  _main
        .p2align        4, 0x90
_main:
        .cfi_startproc
        callq   _a
        retq
        .cfi_endproc

        .p2align        4, 0x90
_a:
        .cfi_startproc
        callq   _b
        retq
        .cfi_endproc

        .p2align        4, 0x90
_b:
        .cfi_startproc
        callq   _c
        retq
        .cfi_endproc

        .p2align        4, 0x90
_c:
        .cfi_startproc
        retq
        .cfi_endproc

// _d should NOT have an entry in .eh_frame
// So it shouldn't be covered by the unwind table.
_d:
        xorl %eax, %eax
        ret

.subsections_via_symbols

#--- cues-range-gap.s
        .text
        .globl  _main
        .p2align        4, 0x90
_main:
        .cfi_startproc
        callq   _a
        retq
        .cfi_endproc

        .p2align        4, 0x90
_a:
        .cfi_startproc
        callq   _b
        retq
        .cfi_endproc

// _d is intentionally placed in between the symbols with eh_frame entries.
// but _d should NOT have an entry in .eh_frame
// So it shouldn't be covered by the unwind table.
_d:
        xorl %eax, %eax
        ret

        .p2align        4, 0x90
_b:
        .cfi_startproc
        callq   _c
        retq
        .cfi_endproc

        .p2align        4, 0x90
_c:
        .cfi_startproc
        retq
        .cfi_endproc

.subsections_via_symbols
