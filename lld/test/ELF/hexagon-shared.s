# REQUIRES: hexagon
# RUN: llvm-mc -filetype=obj -triple=hexagon-unknown-elf %s -o %t
# RUN: llvm-mc -filetype=obj -triple=hexagon-unknown-elf %S/Inputs/hexagon-shared.s -o %t2.o
# RUN: ld.lld -shared %t2.o -soname so -o %t3.so
# RUN: ld.lld -shared %t %t3.so -o %t4.so
# RUN: llvm-objdump -d -j .plt %t4.so | FileCheck --check-prefix=PLT %s
# RUN: llvm-objdump -d -j .text %t4.so | FileCheck --check-prefix=TEXT %s
# RUN: llvm-objdump -D -j .got %t4.so | FileCheck --check-prefix=GOT %s
# RUN: llvm-readelf -r  %t4.so | FileCheck --check-prefix=RELO %s

.global foo
foo:

# _HEX_32_PCREL
.word _DYNAMIC - .
call ##bar

# R_HEX_PLT_B22_PCREL
call bar@PLT

# R_HEX_GOT_11_X and R_HEX_GOT_32_6_X
r2=add(pc,##_GLOBAL_OFFSET_TABLE_@PCREL)
r0 = memw (r2+##bar@GOT)
jumpr r0

# R_HEX_GOT_16_X
r0 = add(r1,##bar@GOT)

# R_HEX_32
.data
.global var
.type var,@object
.p2align 2
var:
   .word 10
   .size var, 4
.global pvar
.type pvar,@object
pvar:
   .word var
   .size pvar, 4

# PLT: { immext(#131200
# PLT: r28 = add(pc,##131252) }
# PLT: { r14 -= add(r28,#16)
# PLT: r15 = memw(r28+#8)
# PLT: r28 = memw(r28+#4) }
# PLT: { r14 = asr(r14,#2)
# PLT: jumpr r28 }
# PLT: { trap0(#219) }
# PLT: immext(#131200)
# PLT: r14 = add(pc,##131236) }
# PLT: r28 = memw(r14+#0) }
# PLT: jumpr r28 }

# TEXT:  10218: 68 00 01 00 00010068
# TEXT: { 	call 0x10270 }
# TEXT: r0 = add(r1,##-65548) }

# GOT: .got:
# GOT: 202f8:	00 00 00 00 00000000 <unknown>

# RELO: 000202f8  00000121 R_HEX_GLOB_DAT
# RELO: 00030300  00000406 R_HEX_32
# RELO: 00030314  00000122 R_HEX_JMP_SLOT
