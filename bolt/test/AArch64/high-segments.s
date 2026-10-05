## Check that BOLT can rewrite AArch64 binaries when a user supplies a custom
## location using the --custom-allocation-vma flag. This is a port of the 
## corresponding X86 test of the same name. 

# RUN: split-file %s %t
# RUN: llvm-mc -filetype=obj -triple=aarch64-unknown-unknown %t/main.s -o %t.o 
# RUN: ld.lld --emit-relocs -T %t/main.ls %t.o -o %t.exe
# RUN: llvm-bolt --custom-allocation-vma=0x700000 \
# RUN:   %t.exe -o %t.bolt 2>&1 | FileCheck %s

# CHECK: BOLT-INFO: creating new program header table at address 0x800000,
# CHECK: BOLT-INFO: enabling relocation mode

//--- main.s
        .type            reserved_space,@object
        .section        .my.reserved.section,"awx",@nobits
        .globl           reserved_space
        .p2align         4, 0x0
reserved_space:
    .zero  0x80000000
    .size   reserved_space, 0x80000000

        .text
        .globl _start
        .type _start, %function
        .reloc 0, R_AARCH64_NONE
_start:
    nop
    nop
    nop
    ret
    .size _start, .-_start

//--- main.ls
SECTIONS
{
    .my.reserved.section 1<<40 : {
      *(.my.reserved.section);
    }
} INSERT BEFORE .comment;