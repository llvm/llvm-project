# REQUIRES: ppc
# RUN: echo 'SECTIONS { \
# RUN:       .text_low 0x2000: { *(.text_low) } \
# RUN:       .text_high 0x2002000 : { *(.text_high) } \
# RUN:       }' > %t.script

# RUN: llvm-mc -filetype=obj -triple=powerpc %s -o %t.o
# RUN: ld.lld -T %t.script %t.o -o %t
# RUN: llvm-readelf -r %t | FileCheck --check-prefix=SEC %s
# RUN: llvm-objdump -d --no-show-raw-insn %t | FileCheck --check-prefixes=CHECK,PD %s
# RUN: llvm-nm --no-sort %t | FileCheck --check-prefix=NM %s

# RUN: ld.lld -T %t.script -pie %t.o -o %t
# RUN: llvm-readelf -r %t | FileCheck --check-prefix=SEC %s
# RUN: llvm-objdump -d --no-show-raw-insn %t | FileCheck --check-prefixes=CHECK,PI %s
# RUN: llvm-nm --no-sort %t | FileCheck --check-prefix=NM %s

# SEC: There are no relocations in this file.

# CHECK:      <_start>:
# CHECK-NEXT:     2000: bl 0x2018
# CHECK-NEXT:           bl 0x2018
# CHECK-NEXT:           bl 0x2018
# CHECK-NEXT:           bl 0x2002008
# PD-NEXT:              bl 0x2028
# PI-NEXT:              bl 0x2038

## high = 0x02002008 = 65536*512+8200
# PD:         <__LongThunk_high>:
# PD-NEXT:        2018: lis 12, 512
# PD-NEXT:              addi 12, 12, 8200
# PD-NEXT:              mtctr 12
# PD-NEXT:              bctr

## .text_high+16 = 0x02002010 = 65536*512+8208
# PD:         <__LongThunk_>:
# PD-NEXT:        2028: lis 12, 512
# PD-NEXT:              addi 12, 12, 8208
# PD-NEXT:              mtctr 12
# PD-NEXT:              bctr

## high-0x2028 = 0x02002008-0x2020 = 65536*512-24
# PI:         <__LongThunk_high>:
# PI-NEXT:        2018: mflr 0
# PI-NEXT:              bcl 20, 31, 0x2020
# PI-NEXT:        2020: mflr 12
# PI-NEXT:              addis 12, 12, 512
# PI-NEXT:              addi 12, 12, -24
# PI-NEXT:              mtlr 0
# PI-NEXT:              mtctr 12
# PI-NEXT:              bctr

## .text_high+16-0x2048 = 0x02002010-0x2048 = 65536*512-48
# PI:         <__LongThunk_>:
# PI-NEXT:        2038: mflr 0
# PI-NEXT:              bcl 20, 31, 0x2040
# PI-NEXT:        2040: mflr 12
# PI-NEXT:              addis 12, 12, 512
# PI-NEXT:              addi 12, 12, -48
# PI-NEXT:              mtlr 0
# PI-NEXT:              mtctr 12
# PI-NEXT:              bctr

.section .text_low, "ax", %progbits
.globl _start
_start:
bl high@local     # Need a thunk
bl high@local     # Need a thunk
bl high+32768@plt # Need a thunk
bl high
bl .text_high+16  # Need a thunk
blr

# PD:         02002008 <high>:
# PD-NEXT:              bl 0x2008
# PD-NEXT:              bl 0x2002010
# PD:         <__LongThunk_>:
# PD-NEXT:     2002010: lis 12, 0
# PD-NEXT:              addi 12, 12, 8200
# PD-NEXT:              mtctr 12
# PD-NEXT:              bctr

.section .text_high, "ax", %progbits
nop
nop
.globl high
high:
bl .text_low+8
bl .text_low+8    # Need a thunk

# NM:      t __LongThunk_high
# NM-NEXT: t __LongThunk_
# NM-NEXT: t __LongThunk_
# NM-NEXT: T _start
# NM-NEXT: T high
