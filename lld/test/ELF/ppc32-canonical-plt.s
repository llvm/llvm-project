# REQUIRES: ppc

## Test that we create canonical PLT entries for -no-pie.

# RUN: llvm-mc -filetype=obj -triple=powerpc %s -o %t.o
# RUN: llvm-mc -filetype=obj -triple=powerpc %p/Inputs/canonical-plt-pcrel.s -o %t1.o
# RUN: ld.lld %t1.o -o %t1.so -shared -soname=so

# RUN: ld.lld %t.o %t1.so -o %t
# RUN: llvm-readobj -r %t | FileCheck --check-prefix=REL %s
# RUN: llvm-readelf -S -s %t | FileCheck --check-prefix=SYM %s
# RUN: llvm-readelf -x .plt %t | FileCheck --check-prefix=HEX %s
# RUN: llvm-objdump -d --no-show-raw-insn %t | FileCheck %s

# REL:      Relocations [
# REL-NEXT:   .rela.plt {
# REL-NEXT:     0x10030318 R_PPC_JMP_SLOT func 0x0
# REL-NEXT:     0x1003031C R_PPC_JMP_SLOT func2 0x0
# REL-NEXT:     0x10030320 R_PPC_JMP_SLOT ifunc 0x0
# REL-NEXT:   }
# REL-NEXT: ]

# SYM: .glink PROGBITS 1001022c

## st_value points to the canonical PLT entry in .glink
# SYM: Symbol table '.dynsym'
# SYM: 1001023c 0 FUNC GLOBAL DEFAULT UND func
# SYM: 1001022c 0 FUNC GLOBAL DEFAULT UND func2
# SYM: 1001024c 0 FUNC GLOBAL DEFAULT UND ifunc
# SYM: Symbol table '.symtab'
# SYM: 1001023c 0 FUNC GLOBAL DEFAULT UND func
# SYM: 1001022c 0 FUNC GLOBAL DEFAULT UND func2
# SYM: 1001024c 0 FUNC GLOBAL DEFAULT UND ifunc

# HEX: 0x10030318 1001025c 10010260 10010264

## Canonical PLT entry of func2.
## 0x1003031C = 65536*4099+796
# CHECK:      1001022c <.glink>:
# CHECK-NEXT:           lis 11, 4099
# CHECK-NEXT:           lwz 11, 796(11)
# CHECK-NEXT:           mtctr 11
# CHECK-NEXT:           bctr

## Canonical PLT entry of func.
## 0x10030318 = 65536*4099+792
# CHECK-NEXT: 1001023c: lis 11, 4099
# CHECK-NEXT:           lwz 11, 792(11)
# CHECK-NEXT:           mtctr 11
# CHECK-NEXT:           bctr

## Canonical PLT entry of ifunc.
## 0x10030320 = 65536*4099+800
# CHECK-NEXT: 1001024c: lis 11, 4099
# CHECK-NEXT:           lwz 11, 800(11)
# CHECK-NEXT:           mtctr 11
# CHECK-NEXT:           bctr

## The 3 b instructions are referenced by .plt entries.
# CHECK-NEXT: 1001025c: b 0x10010268
# CHECK-NEXT:           b 0x10010268
# CHECK-NEXT:           b 0x10010268

## PLTresolve of 64 bytes is at the end.
## Operands of addis & addi: -0x1001025c = 65536*-4097-604
# CHECK-NEXT:           lis 12, 0
# CHECK-NEXT:           addis 11, 11, -4097
# CHECK-NEXT:           lwz 0, 4(12)
# CHECK-NEXT:           addi 11, 11, -604
# CHECK-NEXT:           mtctr 0
# CHECK-NEXT:           add 0, 11, 11
# CHECK-NEXT:           lwz 12, 8(12)
# CHECK-NEXT:           add 11, 0, 11
# CHECK-NEXT:           bctr
# CHECK-COUNT-7:        nop

.globl _start
_start:
  b func
  lis 3, func2@ha
  la 3, func2@l(3)
  lis 3, func@ha
  la 3, func@l(3)
  lis 4, ifunc@ha
  la 4, ifunc@l(4)
