# REQUIRES: riscv
# RUN: echo '.globl bar, weak; .type bar,@function; .type weak,@function; bar: weak:' > %t1.s

# RUN: llvm-mc -filetype=obj -triple=riscv32 %t1.s -o %t1.32.o
# RUN: ld.lld -shared %t1.32.o -soname=t1.32.so -o %t1.32.so
# RUN: llvm-mc -filetype=obj -triple=riscv32 %s -o %t.32.o
# RUN: ld.lld %t.32.o %t1.32.so -z separate-code -o %t.32
# RUN: llvm-readelf -S -s %t.32 | FileCheck --check-prefixes=SEC,NM %s
# RUN: llvm-readobj -r %t.32 | FileCheck --check-prefix=RELOC32 %s
# RUN: llvm-readelf -x .got.plt %t.32 | FileCheck --check-prefix=GOTPLT32 %s
# RUN: llvm-objdump -d --no-show-raw-insn %t.32 | FileCheck --check-prefixes=DIS,DIS32 %s

# RUN: llvm-mc -filetype=obj -triple=riscv64 %t1.s -o %t1.64.o
# RUN: ld.lld -shared %t1.64.o -soname=t1.64.so -o %t1.64.so
# RUN: llvm-mc -filetype=obj -triple=riscv64 %s -o %t.64.o
# RUN: ld.lld %t.64.o %t1.64.so -z separate-code -o %t.64
# RUN: llvm-readelf -S -s %t.64 | FileCheck --check-prefixes=SEC,NM %s
# RUN: llvm-readobj -r %t.64 | FileCheck --check-prefix=RELOC64 %s
# RUN: llvm-readelf -x .got.plt %t.64 | FileCheck --check-prefix=GOTPLT64 %s
# RUN: llvm-objdump -d --no-show-raw-insn %t.64 | FileCheck --check-prefixes=DIS,DIS64 %s

# SEC: .plt PROGBITS {{0*}}00011030

## A canonical PLT has a non-zero st_value. bar and weak are called but their
## addresses are not taken, so a canonical PLT is not necessary.
# NM: {{0*}}00000000 0 FUNC GLOBAL DEFAULT UND bar
# NM: {{0*}}00000000 0 FUNC WEAK   DEFAULT UND weak

## The .got.plt slots relocated by .rela.plt point to .plt
## This is required by glibc.
# RELOC32:      .rela.plt {
# RELOC32-NEXT:   0x13070 R_RISCV_JUMP_SLOT bar 0x0
# RELOC32-NEXT:   0x13074 R_RISCV_JUMP_SLOT weak 0x0
# RELOC32-NEXT: }
# GOTPLT32:      section '.got.plt'
# GOTPLT32-NEXT: 0x00013068 00000000 00000000 30100100 30100100

# RELOC64:      .rela.plt {
# RELOC64-NEXT:   0x130E0 R_RISCV_JUMP_SLOT bar 0x0
# RELOC64-NEXT:   0x130E8 R_RISCV_JUMP_SLOT weak 0x0
# RELOC64-NEXT: }
# GOTPLT64:      section '.got.plt'
# GOTPLT64-NEXT: 0x000130d0 00000000 00000000 00000000 00000000
# GOTPLT64-NEXT: 0x000130e0 30100100 00000000 30100100 00000000

# DIS:      <_start>:
## Direct call
## foo - . = 0x11020-0x11000 = 32
# DIS-NEXT:   11000: auipc ra, 0
# DIS-NEXT:          jalr 32(ra)
## bar@plt - . = 0x11050-0x11008 = 72
# DIS-NEXT:   11008: auipc ra, 0
# DIS-NEXT:          jalr 72(ra)
## bar@plt - . = 0x11050-0x11010 = 64
# DIS-NEXT:   11010: auipc ra, 0
# DIS-NEXT:          jalr 64(ra)
## weak@plt - . = 0x11060-0x11018 = 72
# DIS-NEXT:   11018: auipc ra, 0
# DIS-NEXT:          jalr 72(ra)
# DIS:      <foo>:
# DIS-NEXT:   11020:

# DIS:      Disassembly of section .plt:
# DIS:      <.plt>:
# DIS-NEXT:     auipc t2, 2
# DIS-NEXT:     sub t1, t1, t3
## .got.plt - .plt = 0x13068 - 0x11030 = 4096*2+56
# DIS32-NEXT:   lw t3, 56(t2)
# DIS64-NEXT:   ld t3, 160(t2)
# DIS-NEXT:     addi t1, t1, -44
# DIS32-NEXT:   addi t0, t2, 56
# DIS64-NEXT:   addi t0, t2, 160
# DIS32-NEXT:   srli t1, t1, 2
# DIS64-NEXT:   srli t1, t1, 1
# DIS32-NEXT:   lw t0, 4(t0)
# DIS64-NEXT:   ld t0, 8(t0)
# DIS-NEXT:     jr t3

## 32-bit: &.got.plt[bar]-. = 0x13070-0x11050 = 4096*2+32
# DIS:        11050: auipc t3, 2
# DIS32-NEXT:   lw t3, 32(t3)
# DIS64-NEXT:   ld t3, 144(t3)
# DIS-NEXT:     jalr t1, t3
# DIS-NEXT:     nop

## 32-bit: &.got.plt[weak]-. = 0x13074-0x11060 = 4096*2+20
# DIS:        11060: auipc t3, 2
# DIS32-NEXT:   lw t3, 20(t3)
# DIS64-NEXT:   ld t3, 136(t3)
# DIS-NEXT:     jalr t1, t3
# DIS-NEXT:     nop

.global _start, foo, bar
.weak weak

_start:
  call foo
  call bar
  call bar@plt
  call weak

## foo is local and non-preemptale, no PLT is generated.
foo:
  ret
