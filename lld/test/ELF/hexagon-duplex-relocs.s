# REQUIRES: hexagon
# RUN: llvm-mc -filetype=obj -triple=hexagon-unknown-elf %s -o %t.o
# RUN: llvm-readobj -r %t.o | FileCheck --check-prefix=RELOC %s
# RUN: ld.lld %t.o -o %t
# RUN: llvm-objdump -d --no-show-raw-insn --print-imm-hex %t | FileCheck %s

## Test R_HEX_8_X relocation on duplex instructions (SA1_addi).
## findMaskR8() must use the duplex mask 0x03f00000 when parse bits [15:14]
## are zero.  Pair a non-duplex form with a duplex form using the same symbol
## and verify both resolve to the same address.
##
## Duplex paths for the other findMask functions are already covered:
##   findMaskR6:  hexagon.s (R_HEX_6_X duplex)
##   findMaskR11: hexagon-tls-ie.s (R_HEX_TPREL_11_X duplex)
##   findMaskR16: hexagon-shared.s (R_HEX_GOT_16_X duplex)

	.globl	_start, target
	.type	_start, @function
_start:

# Non-duplex reference (R_HEX_16_X, via findMaskR16)
# RELOC:      R_HEX_32_6_X target 0x0
# RELOC-NEXT: R_HEX_16_X target 0x0
# CHECK:      { immext(#
# CHECK-NEXT:   r0 = add(r0,##[[ADDR:0x[0-9a-f]+]]) }
	r0 = add(r0, ##target)

# Duplex form (R_HEX_8_X, via findMaskR8)
# RELOC-NEXT: R_HEX_32_6_X target 0x0
# RELOC-NEXT: R_HEX_8_X target 0x0
# CHECK-NEXT: { immext(#
# CHECK-NEXT:   r0 = add(r0,##[[ADDR]]); memw(r1+#0x0) = r2 }
	{ r0 = add(r0, ##target)
	  memw(r1+#0) = r2 }

	jumpr r31

target:
	nop
	jumpr r31
