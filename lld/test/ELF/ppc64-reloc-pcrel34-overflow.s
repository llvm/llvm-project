# REQUIRES: ppc
# RUN: echo 'SECTIONS { \
# RUN:       .text_low 0x10010000: { *(.text_low) } \
# RUN:       .text_overflow 0x1000000000 : { *(.text_overflow) } \
# RUN:       }' > %t.script

# RUN: llvm-mc -filetype=obj -triple=powerpc64le %s -o %t.o
# RUN: not ld.lld -T %t.script %t.o -o %t

# RUN: llvm-mc -filetype=obj -triple=powerpc64 %s -o %t.o
# RUN: not ld.lld -T %t.script %t.o -o %t

.section .text_low, "ax", %progbits
# CHECK: relocation R_PPC64_PCREL34 out of range
GlobIntOverflow:
	plwa 3, glob_overflow@PCREL(0), 1
	blr
.section .text_overflow, "ax", %progbits
glob_overflow:
	.long	0
	.size	glob_overflow, 4
