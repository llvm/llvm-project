# REQUIRES: aarch64

## Test that a zero-length symbol at the end of a SHF_MERGE|SHF_STRINGS section
## (offset == section size) is accepted in relocatable output (-r).
## This is valid: label '2:' marks the end of the string and is used to compute
## section size via (2b - 3b). GNU ld handles this case.
## See https://github.com/llvm/llvm-project/issues/118148

# RUN: llvm-mc -filetype=obj -triple=aarch64 %s -o %t.o
# RUN: ld.lld -r %t.o -o %t2.o
# RUN: llvm-readelf -S %t2.o | FileCheck %s --check-prefix=SEC
# RUN: llvm-readelf -s %t2.o | FileCheck %s --check-prefix=SYM
# RUN: llvm-readelf -r %t2.o | FileCheck %s --check-prefix=RELA

## .rodata.str section must be preserved with SHF_MERGE|SHF_STRINGS flags
# SEC: .rodata.str
# SEC-SAME: AMS

## End-of-section symbol .Ltmp1 must be preserved at offset == section size
# SYM: .Ltmp1

## Both relocations in .bug_frames must be present
# RELA: .rodata.str
# RELA: .Ltmp1

    .section .rodata.str,"aMS",%progbits,1
1:
    .asciz "test"
2:

    .section .bug_frames,"a",%progbits
3:
    .p2align 2
    .long (1b - 3b)
    .long (2b - 3b)

    .global test
    .type test, %function
test:
    ret
