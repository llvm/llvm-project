# REQUIRES: ppc
# RUN: llvm-mc -filetype=obj -triple=powerpc %s -o %t.o
# RUN: echo '.globl f, g, h; f: g: h:' | llvm-mc -filetype=obj -triple=powerpc - -o %t1.o
# RUN: ld.lld -shared %t1.o -soname t1.so -o %t1.so

## Check we can create PLT entries for -fno-PIE executable.
# RUN: ld.lld %t.o %t1.so -o %t
# RUN: llvm-readobj -r -d %t | FileCheck --check-prefix=RELOC %s
# RUN: llvm-readelf -S %t | FileCheck --check-prefix=SEC %s
# RUN: llvm-readelf -x .plt %t | FileCheck --check-prefix=HEX %s
# RUN: llvm-objdump -d --no-show-raw-insn %t | FileCheck %s

# SEC:   .got PROGBITS 100202b8
# RELOC: PPC_GOT 0x100202B8

# RELOC:      .rela.plt {
# RELOC-NEXT:   0x100302C4 R_PPC_JMP_SLOT f 0x0
# RELOC-NEXT:   0x100302C8 R_PPC_JMP_SLOT g 0x0
# RELOC-NEXT: }

## .got2+0x8000-0x10004 = 0x30000+0x8000-0x10004 = 65536*2+32764
# CHECK-LABEL: <_start>:
# CHECK-NEXT:    bl 0x100101d0
# CHECK-NEXT:    bl 0x100101d0
# CHECK-NEXT:    bl 0x100101e0
# CHECK-NEXT:    bl 0x100101e0
# CHECK-EMPTY:

## -fno-PIC call stubs of f and g.
## .plt[0] = 0x100302c4 = 65536*4099+708
## .plt[1] = 0x100302c8 = 65536*4099+712
# CHECK-NEXT:  <00000000.plt_call32.f>:
# CHECK-NEXT:    lis 11, 4099
# CHECK-NEXT:    lwz 11, 708(11)
# CHECK-NEXT:    mtctr 11
# CHECK-NEXT:    bctr
# CHECK-EMPTY:
# CHECK-NEXT:  <00000000.plt_call32.g>:
# CHECK-NEXT:    lis 11, 4099
# CHECK-NEXT:    lwz 11, 712(11)
# CHECK-NEXT:    mtctr 11
# CHECK-NEXT:    bctr
# CHECK-EMPTY:

## In Secure PLT ABI, .plt stores function pointers to first instructions of .glink
# HEX: 0x100302c4 10010200 10010204

## These instructions are referenced by .plt entries.
# CHECK: 10010200 <.glink>:
# CHECK-NEXT: b 0x10010208
# CHECK-NEXT: b 0x10010208

## PLTresolve
## Operands of lis & lwz: .got+4 = 0x10020070+4 = 65536*4098+700
## Operands of addis & addi: -.glink = -0x10010200 = 65536*-4097-512
# CHECK-NEXT: lis 12, 4098
# CHECK-NEXT: addis 11, 11, -4097
# CHECK-NEXT: lwz 0, 700(12)
# CHECK-NEXT: addi 11, 11, -512

# CHECK-NEXT: mtctr 0
# CHECK-NEXT: add 0, 11, 11
# CHECK-NEXT: lwz 12, 704(12)
# CHECK-NEXT: add 11, 0, 11
# CHECK-NEXT: bctr

## glibc crti.o references _GLOBAL_OFFSET_TABLE_.
.section .init
  bcl 20, 31, .+4
.L:
  mflr 30
  addis 30, 30, _GLOBAL_OFFSET_TABLE_-.L@ha
  addi 30, 30, _GLOBAL_OFFSET_TABLE_-.L@l

.text
.globl _start
_start:
  bl f
  bl f
  bl g
  bl g
