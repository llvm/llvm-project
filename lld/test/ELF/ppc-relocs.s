# REQUIRES: ppc
# RUN: llvm-mc -filetype=obj -triple=powerpc-unknown-freebsd %s -o %t.o
# RUN: ld.lld %t.o -o %t
# RUN: llvm-objdump -d --no-show-raw-insn %t | FileCheck %s
# RUN: llvm-objdump -s --no-show-raw-insn %t | FileCheck --check-prefix=HEX %s

.section .R_PPC_ADDR16_HA,"ax",@progbits
.globl _start
_start:
  lis 4, msg@ha
msg:
  .string "foo"
  len = . - msg

# CHECK: Disassembly of section .R_PPC_ADDR16_HA:
# CHECK: _start:
# CHECK:    11000:       lis 4, 1
# CHECK: msg:
# CHECK:    11004:       oris 15, 19, 28416

.section .R_PPC_ADDR16_HI,"ax",@progbits
.globl _starti
_starti:
  lis 4,msgi@h
msgi:
  .string "foo"
  leni = . - msgi

# CHECK: Disassembly of section .R_PPC_ADDR16_HI:
# CHECK: _starti:
# CHECK:    11008:       lis 4, 1
# CHECK: msgi:
# CHECK:    1100c:       oris 15, 19, 28416

.section .R_PPC_ADDR16_LO,"ax",@progbits
  addi 4, 4, msg@l
mystr:
  .asciz "blah"
  len = . - mystr

# CHECK: Disassembly of section .R_PPC_ADDR16_LO:
# CHECK: .R_PPC_ADDR16_LO:
# CHECK:    11010:       addi 4, 4, 4100
# CHECK: mystr:
# CHECK:    11014:       ori 12, 19, 24936

.align  2
.section .R_PPC_REL24,"ax",@progbits
.globl .FR_PPC_REL24
.FR_PPC_REL24:
  b .Lfoox
.section .R_PPC_REL24_2,"ax",@progbits
.Lfoox:

# CHECK: Disassembly of section .R_PPC_REL24:
# CHECK: .FR_PPC_REL24:
# CHECK:    1101c:       b .+4

.section .R_PPC_REL14,"ax",@progbits
.globl .FR_PPC_REL14
.FR_PPC_REL14:
  beq .Lfooy
.section .R_PPC_REL14_2,"ax",@progbits
.Lfooy:

# CHECK: Disassembly of section .R_PPC_REL14:
# CHECK: .FR_PPC_REL14:
# CHECK:    11020:       bt 2, .+4

.section .R_PPC_REL32,"ax",@progbits
.globl .FR_PPC_REL32
.FR_PPC_REL32:
  .long .Lfoox3 - .
.section .R_PPC_REL32_2,"ax",@progbits
.Lfoox3:

# HEX:     .R_PPC_REL32:
# HEX-NEXT: 11024 00000004

.section .R_PPC_ADDR32,"ax",@progbits
.globl .FR_PPC_ADDR32
.FR_PPC_ADDR32:
  .long .Lfoox2
.section .R_PPC_ADDR32_2,"ax",@progbits
.Lfoox2:

# HEX:     .R_PPC_ADDR32:
# HEX-NEXT: 11028 0001102c

.align  2
.section .R_PPC_PLTREL24,"ax",@progbits
.globl .R_PPC_PLTREL24
.FR_PPC_PLTREL24:
  b .Lfoox4@PLT
.section .R_PPC_PLTREL24_2,"ax",@progbits
.Lfoox4:

# CHECK: Disassembly of section .R_PPC_PLTREL24:
# CHECK: .R_PPC_PLTREL24:
# CHECK:    1102c:       b .+4
