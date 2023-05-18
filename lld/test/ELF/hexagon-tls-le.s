# REQUIRES: hexagon
# RUN: llvm-mc -filetype=obj -triple=hexagon-unknown-elf %s -o %t.o
# RUN: llvm-readobj -r %t.o | FileCheck --check-prefix=RELOC %s
# RUN: ld.lld %t.o -o %t
# RUN: llvm-objdump -d --print-imm-hex %t | FileCheck %s

	.globl	_start
	.type	_start, @function
_start:
r0 = ugp

# RELOC:      0x4 R_HEX_TPREL_32_6_X a 0x0
# RELOC-NEXT: 0x8 R_HEX_TPREL_16_X a 0x0
# CHECK:      { immext(#0xffffffc0)
# CHECK-NEXT:   r1 = add(r0,##-0x10) }
                r1 = add(r0,##a@TPREL)

# RELOC-NEXT: 0xC R_HEX_TPREL_32_6_X a 0x0
# RELOC-NEXT: 0x10 R_HEX_TPREL_11_X a 0x0
# CHECK:      { immext(#0xffffffc0)
# CHECK-NEXT:   r2 = memw(r0+##-0x10) }
                r2 = memw(r0+##a@TPREL)

# RELOC-NEXT: 0x14 R_HEX_TPREL_HI16 a 0x0
# R_HEX_TPREL_HI16
# CHECK: {      r3.h = #0xffff }
                r3.h = #a@TPREL

# RELOC-NEXT: 0x18 R_HEX_TPREL_LO16 a 0x0
# R_HEX_TPREL_LO16
# CHECK: {      r3.l = #0xfff0 }
                r3.l = #a@TPREL

# RELOC-NEXT: 0x1C R_HEX_TPREL_16 a 0x0
# CHECK: {      r4 = #-0x10 }
                r4 = #a@TPREL

        .section        .tdata,"awT",@progbits
        .globl  a
        .p2align        2
a:
        .word   1
        .size   a, 4

        .globl  b
        .p2align        2
b:
        .word   2
        .size   b, 4

        .globl  c
        .p2align        2
c:
        .word   3
        .size   c, 4

        .globl  d
        .p2align        2
d:
        .word   4
        .size   d, 4
