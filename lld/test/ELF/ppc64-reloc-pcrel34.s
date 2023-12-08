# REQUIRES: ppc
# RUN: echo 'SECTIONS { \
# RUN:       .text_low 0x10010000: { *(.text_low) } \
# RUN:       .text_high 0x10080000 : { *(.text_high) } \
# RUN:       }' > %t.script

# RUN: llvm-mc -filetype=obj -triple=powerpc64le %s -o %t.o
# RUN: ld.lld -T %t.script %t.o -o %t
# RUN: llvm-readelf -s %t | FileCheck %s --check-prefix=SYMBOL
# RUN: llvm-objdump -d --no-show-raw-insn %t | FileCheck %s

# RUN: llvm-mc -filetype=obj -triple=powerpc64 %s -o %t.o
# RUN: ld.lld -T %t.script %t.o -o %t
# RUN: llvm-readelf -s %t | FileCheck %s --check-prefix=SYMBOL
# RUN: llvm-objdump -d --no-show-raw-insn %t | FileCheck %s

.section .text_low, "ax", %progbits
# CHECK-LABEL: <GlobIntPCRel>:
# CHECK-NEXT:    10010000:        plwa 3, 12(0), 1
# SYMBOL:        1001000c     4 NOTYPE  LOCAL  DEFAULT     1 glob_int
GlobIntPCRel:
	plwa 3, glob_int@PCREL(0), 1
	blr
glob_int:
	.long	0
	.size	glob_int, 4


# CHECK-LABEL: <GlobIntPCRelOffset>:
# CHECK-NEXT:    10010010:        plwa 3, 16(0), 1
# SYMBOL:        1001001c     8 NOTYPE  LOCAL  DEFAULT     1 glob_int8
GlobIntPCRelOffset:
	plwa 3, glob_int8@PCREL+4(0), 1
	blr
glob_int8:
	.quad	0
	.size	glob_int8, 8


# CHECK-LABEL: <GlobIntPCRelBigOffset>:
# CHECK-NEXT:    10010024:        plwa 3, 458720(0), 1
# SYMBOL:        10080000     8 NOTYPE  LOCAL  DEFAULT     2 glob_int8_big
GlobIntPCRelBigOffset:
	plwa 3, glob_int8_big@PCREL+4(0), 1
	blr
.section .text_high, "ax", %progbits
glob_int8_big:
	.quad	0
	.size	glob_int8_big, 8
