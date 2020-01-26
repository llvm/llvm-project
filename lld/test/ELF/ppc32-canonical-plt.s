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
# REL-NEXT:     R_PPC_JMP_SLOT func 0x0
# REL-NEXT:     R_PPC_JMP_SLOT ifunc 0x0
# REL-NEXT:   }
# REL-NEXT: ]

# SYM: .glink PROGBITS 100101dc

## st_value points to the canonical PLT entry in .glink
# SYM: Symbol table '.dynsym'
# SYM: 100101dc 0 FUNC GLOBAL DEFAULT UND func
# SYM: 100101ec 0 FUNC GLOBAL DEFAULT UND ifunc
# SYM: Symbol table '.symtab'
# SYM: 100101dc 0 FUNC GLOBAL DEFAULT UND func
# SYM: 100101ec 0 FUNC GLOBAL DEFAULT UND ifunc

# HEX: 0x100302b4 100101fc 10010200

## Canonical PLT entry of func.
## 0x100101dc + 4*2 + 64 = 0x10010224
## 0x1001021c = 65536*4099+692
# CHECK:      100101dc .glink:
# CHECK-NEXT:           lis 11, 4099
# CHECK-NEXT:           lwz 11, 692(11)
# CHECK-NEXT:           mtctr 11
# CHECK-NEXT:           bctr

## Canonical PLT entry of ifunc.
## 0x10010220 = 65536*4099+696
# CHECK-NEXT: 100101ec: lis 11, 4099
# CHECK-NEXT:           lwz 11, 696(11)
# CHECK-NEXT:           mtctr 11
# CHECK-NEXT:           bctr

## The 2 b instructions are referenced by .plt entries.
# CHECK-NEXT: 100101fc: b .+8
# CHECK-NEXT:           b .+4

## PLTresolve of 64 bytes is at the end.
## Operands of addis & addi: -0x100101fc = 65536*-4097-508
# CHECK-NEXT:           lis 12, 0
# CHECK-NEXT:           addis 11, 11, -4097
# CHECK-NEXT:           lwz 0, 4(12)
# CHECK-NEXT:           addi 11, 11, -508
# CHECK-NEXT:           mtctr 0
# CHECK-NEXT:           add 0, 11, 11
# CHECK-NEXT:           lwz 12, 8(12)
# CHECK-NEXT:           add 11, 0, 11
# CHECK-NEXT:           bctr
# CHECK-COUNT-7:        nop

.globl _start
_start:
  lis 3, func@ha
  la 3, func@l(3)
  lis 4, ifunc@ha
  la 4, ifunc@l(4)
