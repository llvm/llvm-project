# REQUIRES: hexagon
# RUN: llvm-mc -filetype=obj -triple=hexagon-unknown-elf %s -o %t.o
# RUN: llvm-mc -filetype=obj -triple=hexagon-unknown-elf %S/Inputs/hexagon.s -o %t1.o
# RUN: ld.lld %t.o %t1.o -o %t
# RUN: llvm-objdump -d %t | FileCheck %s

# Note: 131584 == 0x20200
# R_HEX_32_6_X
# R_HEX_12_X
if (p0) r0 = ##_start
# CHECK: immext(#131584)
# CHECK: if (p0) r0 = ##131644

# R_HEX_B15_PCREL
if (p0) jump:nt #_start
# CHECK: if (p0) jump:nt 0x2023c

# R_HEX_B32_PCREL_X
# R_HEX_B15_PCREL_X
if (p0) jump:nt ##_start
# CHECK: if (p0) jump:nt 0x2023c

# R_HEX_B22_PCREL
call #_start
# CHECK: call 0x2023c

# R_HEX_B32_PCREL_X
# R_HEX_B22_PCREL_X
call ##_start
# CHECK: immext(#320)
# CHECK: call 0x2023c

# R_HEX_6_X tests:
# One test for each mask in the lookup table.

#0x38000000
if (!P0) memw(r0+#8)=##_start
# CHECK: 38c0e11c   	if (!p0) memw(r0+#8) = ##131644 }

#0x39000000
{ p0 = p1
  if (!P0.new) memw(r0+#0)=##_start }
# CHECK: 39c0e01c   	if (!p0.new) memw(r0+#0) = ##131644 }

#0x3e000000
memw(r0+##_start)+=r1
# CHECK: 3e40de01   	memw(r0+##131644) += r1 }

#0x3f000000
memw(r0+##_start)+=#4
# CHECK: 3f40de04   	memw(r0+##131644) += #4 }

#0x40000000
{ r0 = r1
  if (p0) memb(r0+##_start)=r0.new }
# CHECK: 40a0e2e0   	if (p0) memb(r0+##131644) = r0.new }

#0x41000000
if (p0) r0=memb(r1+##_start)
# CHECK: 4101c780   	if (p0) r0 = memb(r1+##131644) }

#0x42000000
{ r0 = r1
  p0 = p1
  if (p0.new) memb(r0+##_start)=r0.new }
# CHECK: 42a0e2e0   	if (p0.new) memb(r0+##131644) = r0.new }

#0x43000000
{ p0 = p1
 if (P0.new) r0=memb(r0+##_start) }
# CHECK: 4300c780   	if (p0.new) r0 = memb(r0+##131644) }

#0x44000000
if (!p0) memb(r0+##_start)=r1
# CHECK: 4400e1e0   	if (!p0) memb(r0+##131644) = r1 }

#0x45000000
if (!p0) r0=memb(r1+##_start)
# CHECK: 4501c780   	if (!p0) r0 = memb(r1+##131644) }

#0x46000000
{ p0 = p1
  if (!p0.new) memb(r0+##_start)=r1 }
# CHECK: 4600e1e0   	if (!p0.new) memb(r0+##131644) = r1 }

#0x47000000
{ p0 = p1
  if (!p0.new) r0=memb(r1+##_start) }
# CHECK: 4701c780   	if (!p0.new) r0 = memb(r1+##131644) }

#0x6a000000 -- Note 4294967132 == -0xa4 the distance between
#              here and _start, so this will change if
#              tests are added between here and _start
r0=add(pc,##_start@pcrel)
# CHECK: 6a49d600  	r0 = add(pc,##236) }

#0x7c000000
r1:0=combine(#8,##_start)
# CHECK: 7c9ec100   	r1:0 = combine(#8,##131644) }

#0x9a000000
r1:0=memb_fifo(r2=##_start)
# CHECK: 9a82df00   	r1:0 = memb_fifo(r2=##131644) }

#0x9b000000
r0=memb(r1=##_start)
# CHECK: 9b01df00   	r0 = memb(r1=##131644) }

#0x9c000000
r1:0=memb_fifo(r2<<#2+##_start)
# CHECK: 9c82ff00   	r1:0 = memb_fifo(r2<<#2+##131644) }

#0x9d000000
r0=memb(r1<<#2+##_start)
# CHECK: 9d01ff00   	r0 = memb(r1<<#2+##131644) }

#0x9f000000
if (!p0) r0=memb(##_start)
# CHECK: 9f1ee880   	if (!p0) r0 = memb(##131644) }

#0xab000000
memb(r0=##_start)=r1
# CHECK: ab00c1bc   	memb(r0=##131644) = r1 }

#0xad000000
memb(r0<<#2+##_start)=r1
# CHECK: ad00e1bc   	memb(r0<<#2+##131644) = r1 }

#0xaf000000
if (!p0) memb(##_start)=r1
# CHECK: af03c1e4   	if (!p0) memb(##131644) = r1 }

#0xd7000000
r0=add(##_start,mpyi(r1,r2))
# CHECK: d761e280   	r0 = add(##131644,mpyi(r1,r2)) }

#0xd8000000
R0=add(##_start,mpyi(r0,#2))
# CHECK: d860e082   	r0 = add(##131644,mpyi(r0,#2)) }

#0xdb000000
r0=add(r1,add(r2,##_start))
# CHECK: db61e082   	r0 = add(r1,add(r2,##131644)) }

#0xdf000000
r0=add(r1,mpyi(r2,##_start))
# CHECK: dfe2e081   	r0 = add(r1,mpyi(r2,##131644)) }

# Duplex form of R_HEX_6_X
# R_HEX_32_6_X
# R_HEX_6_X
{ r0 = ##_start; r2 = r16 }
# CHECK: 2bc03082   	r0 = ##131644; 	r2 = r16 }

# R_HEX_HI16
r0.h = #HI(_start)
# CHECK: r0.h = #2

# R_HEX_LO16
r0.l = #LO(_start)
# CHECK: r0.l = #572

# R_HEX_8_X has 3 relocation mask variations
#0xde000000
r0=sub(##_start, asl(r0, #1))
# CHECK: de20e1c6      r0 = sub(##131644,asl(r0,#1)) }

#0x3c000000
memw(r0+#0) = ##_start
# CHECK: 3c40c03c   	memw(r0+#0) = ##131644 }

# The rest:
r1:0=combine(r2,##_start);
# CHECK: 7302e780   	r1:0 = combine(r2,##131644) }

# R_HEX_32:
r_hex_32:
.word _start
# CHECK: 0002023c

# R_HEX_16_X has 4 relocation mask variations
# 0x48000000
memw(##_start) = r0
# CHECK: 4880c03c   memw(##131644) = r0 }

# 0x49000000
r0 = memw(##_start)
# CHECK: 4980c780   r0 = memw(##131644)

# 0x78000000
r0 = ##_start
# CHECK: 7800c780   r0 = ##131644 }

# 0xb0000000
r0 = add(r1, ##_start)
# CHECK: b001c780   r0 = add(r1,##131644) }

# R_HEX_B9_PCREL:
{r0=#1 ; jump #_start}
# CHECK: jump 0x2023c

# R_HEX_B9_PCREL_X:
{r0=#1 ; jump ##_start}
# CHECK: jump 0x2023c

# R_HEX_B13_PCREL
if (r0 == #0) jump:t #_start
# CHECK: if (r0==#0) jump:t 0x2023c

# R_HEX_9_X
p0 = !cmp.gtu(r0, ##_start)
# CHECK: p0 = !cmp.gtu(r0,##131644)

# R_HEX_10_X
p0 = !cmp.gt(r0, ##_start)
# CHECK: p0 = !cmp.gt(r0,##131644)

# R_HEX_11_X
r0 = memw(r1+##_start)
# CHECK: r0 = memw(r1+##131644)

memw(r0+##_start) = r1
# CHECK: memw(r0+##131644) = r1
