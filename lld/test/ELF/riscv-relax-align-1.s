# REQUIRES: riscv
## Testing the aligment is correct when mixing with rvc/norvc relax/norelax

# RUN: rm -rf %t && split-file %s %t && cd %t

## NORVC, NORELAX
# RUN: llvm-mc -filetype=obj -triple=riscv64 -mattr=+relax,+c,+m a.s -o a.o --defsym=NORVC=1 --defsym=NORELAX=1
# RUN: ld.lld -T lds a.o -o a.out
# RUN: llvm-nm a.out | FileCheck %s --check-prefix=NORVC-NORELAX

# NORVC-NORELAX: 0000000000001010 t SHOULD_ALIGN_8_HERE

## NORVC, RELAX
# RUN: llvm-mc -filetype=obj -triple=riscv64 -mattr=+relax,+c,+m a.s -o a.o --defsym=NORVC=1
# RUN: ld.lld -T lds a.o -o a.out
# RUN: llvm-nm a.out | FileCheck %s --check-prefix=NORVC

# NORVC: 0000000000001010 t SHOULD_ALIGN_8_HERE

## RVC, NORELAX
# RUN: llvm-mc -filetype=obj -triple=riscv64 -mattr=+relax,+c,+m a.s -o a.o --defsym=NORELAX=1
# RUN: ld.lld -T lds a.o -o a.out
# RUN: llvm-nm a.out | FileCheck %s --check-prefix=NORELAX

# NORELAX: 0000000000001010 t SHOULD_ALIGN_8_HERE

## RVC, RELAX
# RUN: llvm-mc -filetype=obj -triple=riscv64 -mattr=+relax,+c,+m a.s -o a.o
# RUN: ld.lld -T lds a.o -o a.out
# RUN: llvm-nm a.out | FileCheck %s --check-prefix=RELAX-RVC

# RELAX-RVC: 0000000000001010 t SHOULD_ALIGN_8_HERE

#--- a.s
        .text
        .option relax
        .balign 4
        .global _start
        .type _start, @function
_start:
        lui a0, %hi(foo)
        addi a0, a0, %lo(foo)
	mul a0, a1, a4
        .option push

.ifdef NORELAX
        .option norelax
.endif
.ifdef NORVC
        .option norvc
.endif
        .balign 8
SHOULD_ALIGN_8_HERE:
        .word 0x12345678

        .option pop

foo:
        ret



#--- lds
ENTRY(_start)
SECTIONS {
	.text 0x0001000 : {
		*(.text*)
	}
}
