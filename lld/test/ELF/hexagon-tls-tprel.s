# REQUIRES: hexagon
# RUN: llvm-mc -filetype=obj -triple=hexagon-unknown-elf %s -o %t.o
# RUN: llvm-readobj -r %t.o | FileCheck --check-prefix=RELOC %s
# RUN: ld.lld %t.o -o %t
# RUN: llvm-objdump -d --print-imm-hex %t | FileCheck %s

# R_HEX_6_X@TPREL tests:
# One test for each mask in the lookup table.

#0x38000000
if (!P0) memw(r0+#8)=##a@TPREL
# RELOC: 0x0 R_HEX_TPREL_32_6_X a 0x0
# RELOC-NEXT: 0x4 R_HEX_TPREL_16_X a 0x0
# CHECK:  { 	immext(#0xffffffc0)
# CHECK-NEXT:    	if (!p0) memw(r0+#0x8) = ##-0x4 }

#0x39000000
{ p0 = p1
  if (!P0.new) memw(r0+#0)=##a@TPREL }
# RELOC-NEXT: 0xC R_HEX_TPREL_32_6_X a 0x0
# RELOC-NEXT: 0x10 R_HEX_TPREL_16_X a 0x0
# CHECK:    	immext(#0xffffffc0)
# CHECK-NEXT:    	if (!p0.new) memw(r0+#0x0) = ##-0x4 }

#0x3e000000
memw(r0+##a@TPREL)+=r1
# RELOC-NEXT: 0x14 R_HEX_TPREL_32_6_X a 0x0
# RELOC-NEXT: 0x18 R_HEX_TPREL_16_X a 0x0
# CHECK:  { 	immext(#0xffffffc0)
# CHECK-NEXT:    	memw(r0+##0xfffffffc) += r1 }

#0x3f000000
memw(r0+##a@TPREL)+=#4
# RELOC-NEXT: 0x1C R_HEX_TPREL_32_6_X a 0x0
# RELOC-NEXT: 0x20 R_HEX_TPREL_16_X a 0x0
# CHECK:  { 	immext(#0xffffffc0)
# CHECK-NEXT:    	memw(r0+##0xfffffffc) += #0x4 }


#0x40000000
{ r0 = r1
  if (p0) memb(r0+##a@TPREL)=r0.new }
# RELOC-NEXT: 0x28 R_HEX_TPREL_32_6_X a 0x0
# RELOC-NEXT: 0x2C R_HEX_TPREL_16_X a 0x0
# CHECK:    	immext(#0xffffffc0)
# CHECK-NEXT:    	if (p0) memb(r0+##0xfffffffc) = r0.new }

#0x41000000
if (p0) r0=memb(r1+##a@TPREL)
# RELOC-NEXT: 0x30 R_HEX_TPREL_32_6_X a 0x0
# RELOC-NEXT: 0x34 R_HEX_TPREL_16_X a 0x0
# CHECK:  { 	immext(#0xffffffc0)
# CHECK-NEXT:    	if (p0) r0 = memb(r1+##0xfffffffc) }

#0x42000000
{ r0 = r1
  p0 = p1
  if (p0.new) memb(r0+##a@TPREL)=r0.new }
# RELOC-NEXT: 0x40 R_HEX_TPREL_32_6_X a 0x0
# RELOC-NEXT: 0x44 R_HEX_TPREL_16_X a 0x0
# CHECK:    	immext(#0xffffffc0)
# CHECK-NEXT:    	if (p0.new) memb(r0+##0xfffffffc) = r0.new }

#0x43000000
{ p0 = p1
 if (P0.new) r0=memb(r0+##a@TPREL) }
# RELOC-NEXT: 0x4C R_HEX_TPREL_32_6_X a 0x0
# RELOC-NEXT: 0x50 R_HEX_TPREL_16_X a 0x0
# CHECK:    	immext(#0xffffffc0)
# CHECK-NEXT:    	if (p0.new) r0 = memb(r0+##0xfffffffc) }

#0x44000000
if (!p0) memb(r0+##a@TPREL)=r1
# RELOC-NEXT: 0x54 R_HEX_TPREL_32_6_X a 0x0
# RELOC-NEXT: 0x58 R_HEX_TPREL_16_X a 0x0
# CHECK:  { 	immext(#0xffffffc0)
# CHECK-NEXT:    	if (!p0) memb(r0+##0xfffffffc) = r1 }

#0x45000000
if (!p0) r0=memb(r1+##a@TPREL)
# RELOC-NEXT: 0x5C R_HEX_TPREL_32_6_X a 0x0
# RELOC-NEXT: 0x60 R_HEX_TPREL_16_X a 0x0
# CHECK:  { 	immext(#0xffffffc0)
# CHECK-NEXT:    	if (!p0) r0 = memb(r1+##0xfffffffc) }


#0x46000000
{ p0 = p1
  if (!p0.new) memb(r0+##a@TPREL)=r1 }
# RELOC-NEXT: 0x68 R_HEX_TPREL_32_6_X a 0x0
# RELOC-NEXT: 0x6C R_HEX_TPREL_16_X a 0x0
# CHECK:    	immext(#0xffffffc0)
# CHECK-NEXT:    	if (!p0.new) memb(r0+##0xfffffffc) = r1 }

#0x47000000
{ p0 = p1
  if (!p0.new) r0=memb(r1+##a@TPREL) }
# RELOC-NEXT: 0x74 R_HEX_TPREL_32_6_X a 0x0
# RELOC-NEXT: 0x78 R_HEX_TPREL_16_X a 0x0
# CHECK:    	immext(#0xffffffc0)
# CHECK-NEXT:    	if (!p0.new) r0 = memb(r1+##0xfffffffc) }

#0x7c000000
r1:0=combine(#8,##a@TPREL)
# RELOC-NEXT: 0x7C R_HEX_TPREL_32_6_X a 0x0
# RELOC-NEXT: 0x80 R_HEX_TPREL_16_X a 0x0
# CHECK:  { 	immext(#0xffffffc0)
# CHECK-NEXT:    	r1:0 = combine(#0x8,##0xfffffffc) }


#0x9a000000
r1:0=memb_fifo(r2=##a@TPREL)
# RELOC-NEXT: 0x84 R_HEX_TPREL_32_6_X a 0x0
# RELOC-NEXT: 0x88 R_HEX_TPREL_16_X a 0x0
# CHECK:  { 	immext(#0xffffffc0)
# CHECK-NEXT:    	r1:0 = memb_fifo(r2=##0xfffffffc) }


#0x9b000000
r0=memb(r1=##a@TPREL)
# RELOC-NEXT: 0x8C R_HEX_TPREL_32_6_X a 0x0
# RELOC-NEXT: 0x90 R_HEX_TPREL_16_X a 0x0
# CHECK:  { 	immext(#0xffffffc0)
# CHECK-NEXT:    	r0 = memb(r1=##0xfffffffc) }


#0x9c000000
r1:0=memb_fifo(r2<<#2+##a@TPREL)
# RELOC-NEXT: 0x94 R_HEX_TPREL_32_6_X a 0x0
# RELOC-NEXT: 0x98 R_HEX_TPREL_16_X a 0x0
# CHECK:  { 	immext(#0xffffffc0)
# CHECK-NEXT:    	r1:0 = memb_fifo(r2<<#0x2+##0xfffffffc) }


#0x9d000000
r0=memb(r1<<#2+##a@TPREL)
# RELOC-NEXT: 0x9C R_HEX_TPREL_32_6_X a 0x0
# RELOC-NEXT: 0xA0 R_HEX_TPREL_16_X a 0x0
# CHECK:  { 	immext(#0xffffffc0)
# CHECK-NEXT:    	r0 = memb(r1<<#0x2+##0xfffffffc) }


#0x9f000000
if (!p0) r0=memb(##a@TPREL)
# RELOC-NEXT: 0xA4 R_HEX_TPREL_32_6_X a 0x0
# RELOC-NEXT: 0xA8 R_HEX_TPREL_16_X a 0x0
# CHECK:  { 	immext(#0xffffffc0)
# CHECK-NEXT:    	if (!p0) r0 = memb(##0xfffffffc) }


#0xab000000
memb(r0=##a@TPREL)=r1
# RELOC-NEXT: 0xAC R_HEX_TPREL_32_6_X a 0x0
# RELOC-NEXT: 0xB0 R_HEX_TPREL_16_X a 0x0
# CHECK:  { 	immext(#0xffffffc0)
# CHECK-NEXT:    	memb(r0=##0xfffffffc) = r1 }


#0xad000000
memb(r0<<#2+##a@TPREL)=r1
# RELOC-NEXT: 0xB4 R_HEX_TPREL_32_6_X a 0x0
# RELOC-NEXT: 0xB8 R_HEX_TPREL_16_X a 0x0
# CHECK:  { 	immext(#0xffffffc0)
# CHECK-NEXT:    	memb(r0<<#0x2+##0xfffffffc) = r1 }


#0xaf000000
if (!p0) memb(##a@TPREL)=r1
# RELOC-NEXT: 0xBC R_HEX_TPREL_32_6_X a 0x0
# RELOC-NEXT: 0xC0 R_HEX_TPREL_16_X a 0x0
# CHECK:  { 	immext(#0xffffffc0)
# CHECK-NEXT:    	if (!p0) memb(##0xfffffffc) = r1 }


#0xd7000000
r0=add(##a@TPREL,mpyi(r1,r2))
# RELOC-NEXT: 0xC4 R_HEX_TPREL_32_6_X a 0x0
# RELOC-NEXT: 0xC8 R_HEX_TPREL_16_X a 0x0
# CHECK:  { 	immext(#0xffffffc0)
# CHECK-NEXT:    	r0 = add(##0xfffffffc,mpyi(r1,r2)) }


#0xd8000000
R0=add(##a@TPREL,mpyi(r0,#2))
# RELOC-NEXT: 0xCC R_HEX_TPREL_32_6_X a 0x0
# RELOC-NEXT: 0xD0 R_HEX_TPREL_16_X a 0x0
# CHECK:  { 	immext(#0xffffffc0)
# CHECK-NEXT:    	r0 = add(##0xfffffffc,mpyi(r0,#0x2)) }


#0xdb000000
r0=add(r1,add(r2,##a@TPREL))
# RELOC-NEXT: 0xD4 R_HEX_TPREL_32_6_X a 0x0
# RELOC-NEXT: 0xD8 R_HEX_TPREL_16_X a 0x0
# CHECK:  { 	immext(#0xffffffc0)
# CHECK-NEXT:    	r0 = add(r1,add(r2,##-0x4)) }


#0xdf000000
r0=add(r1,mpyi(r2,##a@TPREL))
# RELOC-NEXT: 0xDC R_HEX_TPREL_32_6_X a 0x0
# RELOC-NEXT: 0xE0 R_HEX_TPREL_16_X a 0x0
# CHECK:  { 	immext(#0xffffffc0)
# CHECK-NEXT:    	r0 = add(r1,mpyi(r2,##0xfffffffc)) }


# Duplex form of R_HEX_6_X
# R_HEX_32_6_X
# R_HEX_6_X
{ r0 = ##a@TPREL; r2 = r16 }
# RELOC-NEXT: 0xE4 R_HEX_TPREL_32_6_X a 0x0
# RELOC-NEXT: 0xE8 R_HEX_TPREL_16_X a 0x0
# CHECK:  { 	immext(#0xffffffc0)
# CHECK-NEXT:    	r0 = ##0xfffffffc; 	r2 = r16 }

        .section        .tdata,"awT",@progbits
        .globl  a
        .p2align        2
a:
        .word   1
        .size   a, 4
