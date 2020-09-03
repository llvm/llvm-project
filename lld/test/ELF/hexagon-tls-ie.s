# REQUIRES: hexagon
# RUN: llvm-mc -filetype=obj -triple=hexagon-unknown-elf %s -o %t.o
# RUN: llvm-readobj -r %t.o | FileCheck -check-prefix=RELOC %s
# RUN: ld.lld %t.o -o %t
## shared needs -z notext because of the R_HEX_IE_16/32_X(R_GOT) static
## relocations
# RUN: ld.lld -z notext -shared %t.o -o %t.so
# RUN: llvm-objdump -d --no-show-raw-insn --print-imm-hex %t | FileCheck %s
# RUN:  llvm-readobj -x .got %t | FileCheck -check-prefix=GOT %s
# RUN: llvm-objdump -d --no-show-raw-insn --print-imm-hex %t.so | \
# RUN: FileCheck -check-prefix=SHARED %s
# RUN: llvm-readobj -r  %t.so | FileCheck -check-prefix=RELA %s

	.globl	_start
	.type	_start, @function
_start:

# RELOC:      0x0 R_HEX_IE_32_6_X a 0x0
# RELOC-NEXT: 0x4 R_HEX_IE_16_X a 0x0
# CHECK:      {   immext(#0x30180)
# CHECK-NEXT:     r2 = memw(##0x301a4) }
                  r2 = memw(##a@IE)

# RELOC-NEXT: 0x8 R_HEX_IE_LO16 a 0x0
# CHECK: {       r2.l = #0x1a4 }
                 r2.l = #a@IE
# RELOC-NEXT: 0xC R_HEX_IE_HI16 a 0x0
# CHECK: {       r2.h = #0x3 }
                 r2.h = #a@IE


# GOT: Hex dump of section '.got':
# GOT-NEXT: 0x000301a4 f0ffffff f4ffffff f8ffffff fcffffff
                 r2 = memw(##a@IE)
                 r2 = memw(##b@IE)
                 r2 = memw(##c@IE)
                 r2 = memw(##d@IE)

# RELOC:      0x30 R_HEX_IE_GOT_32_6_X a 0x0
# RELOC-NEXT: 0x34 R_HEX_IE_GOT_16_X a 0x0
# SHARED:      { immext(#0xfffeffc0)
# SHARED-NEXT:   r2 = memw(##0xfffefff0) }
                 r2 = memw(##a@IEGOT)

# RELOC-NEXT: 0x38 R_HEX_IE_GOT_LO16 a 0x0
# SHARED: {     r2.l = #0xfff0 }
                r2.l = #a@IEGOT
# RELOC-NEXT: 0x3C R_HEX_IE_GOT_HI16 a 0x0
# SHARED: {     r2.h = #0xfffe }
                r2.h = #a@IEGOT

# RELOC:      0x44 R_HEX_IE_GOT_11_X a 0x0
# SHARED:    {  immext(#0xfffeffc0)
# SHARED-NEXT:  r0 = !cmp.eq(r1,##-0x10010) }
                r0=!cmp.eq(r1,##a@iegot)

# RELA:       0x203C4 R_HEX_TPREL_32 a 0x0
# RELA-NEXT:  0x203C8 R_HEX_TPREL_32 b 0x0
# RELA-NEXT:  0x203CC R_HEX_TPREL_32 c 0x0
# RELA-NEXT:  0x203D0 R_HEX_TPREL_32 d 0x0
                r2 = memw(##b@IEGOT)
                r2 = memw(##c@IEGOT)
                r2 = memw(##d@IEGOT)


.section        .tdata,"awT",@progbits
.globl  a
a:
.word 1
.globl  b
b:
.word 2
.globl  c
c:
.word 3
.globl  d
d:
.word 4
