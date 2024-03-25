# REQUIRES: loongarch
# RUN: echo '.globl bar, weak; .type bar,@function; .type weak,@function; bar: weak:' > %t1.s

# RUN: llvm-mc --filetype=obj --triple=loongarch32 %t1.s -o %t1.32.o
# RUN: ld.lld -shared %t1.32.o -soname=t1.32.so -o %t1.32.so
# RUN: llvm-mc --filetype=obj --triple=loongarch32 %s -o %t.32.o
# RUN: ld.lld %t.32.o %t1.32.so -z separate-code -o %t.32
# RUN: llvm-readelf -S -s %t.32 | FileCheck --check-prefixes=SEC,NM %s
# RUN: llvm-readobj -r %t.32 | FileCheck --check-prefix=RELOC32 %s
# RUN: llvm-readelf -x .got.plt %t.32 | FileCheck --check-prefix=GOTPLT32 %s
# RUN: llvm-objdump -d --no-show-raw-insn %t.32 | FileCheck --check-prefixes=DIS,DIS32 %s

# RUN: llvm-mc --filetype=obj --triple=loongarch64 %t1.s -o %t1.64.o
# RUN: ld.lld -shared %t1.64.o -soname=t1.64.so -o %t1.64.so
# RUN: llvm-mc --filetype=obj --triple=loongarch64 %s -o %t.64.o
# RUN: ld.lld %t.64.o %t1.64.so -z separate-code -o %t.64
# RUN: llvm-readelf -S -s %t.64 | FileCheck --check-prefixes=SEC,NM %s
# RUN: llvm-readobj -r %t.64 | FileCheck --check-prefix=RELOC64 %s
# RUN: llvm-readelf -x .got.plt %t.64 | FileCheck --check-prefix=GOTPLT64 %s
# RUN: llvm-objdump -d --no-show-raw-insn %t.64 | FileCheck --check-prefixes=DIS,DIS64 %s

# SEC: .plt PROGBITS {{0*}}00020020

## A canonical PLT has a non-zero st_value. bar and weak are called but their
## addresses are not taken, so a canonical PLT is not necessary.
# NM: {{0*}}00000000 0 FUNC GLOBAL DEFAULT UND bar
# NM: {{0*}}00000000 0 FUNC WEAK   DEFAULT UND weak

## The .got.plt slots relocated by .rela.plt point to .plt
## This is required by glibc.
# RELOC32:      .rela.plt {
# RELOC32-NEXT:   0x40070 R_LARCH_JUMP_SLOT bar 0x0
# RELOC32-NEXT:   0x40074 R_LARCH_JUMP_SLOT weak 0x0
# RELOC32-NEXT: }
# GOTPLT32:      section '.got.plt'
# GOTPLT32-NEXT: 0x00040068 00000000 00000000 20000200 20000200

# RELOC64:      .rela.plt {
# RELOC64-NEXT:   0x400E0 R_LARCH_JUMP_SLOT bar 0x0
# RELOC64-NEXT:   0x400E8 R_LARCH_JUMP_SLOT weak 0x0
# RELOC64-NEXT: }
# GOTPLT64:      section '.got.plt'
# GOTPLT64-NEXT: 0x000400d0 00000000 00000000 00000000 00000000
# GOTPLT64-NEXT: 0x000400e0 20000200 00000000 20000200 00000000

# DIS:      <_start>:
## Direct call
## foo - . = 0x20010-0x20000 = 16
# DIS-NEXT:   20000: bl 16
## bar@plt - . = 0x20040-0x20004 = 60
# DIS-NEXT:   20004: bl 60
## bar@plt - . = 0x20040-0x20008 = 56
# DIS-NEXT:   20008: bl 56
## weak@plt - . = 0x20050-0x2000c = 68
# DIS-NEXT:   2000c: bl 68
# DIS:      <foo>:
# DIS-NEXT:   20010:

# DIS:      Disassembly of section .plt:
# DIS:      <.plt>:
## 32-bit: .got.plt - .plt = 0x40068 - 0x20020 = 4096*32+72
# DIS32-NEXT:   pcaddu12i $t2, 32
# DIS32-NEXT:   sub.w     $t1, $t1, $t3
# DIS32-NEXT:   ld.w      $t3, $t2, 72
# DIS32-NEXT:   addi.w    $t1, $t1, -44
# DIS32-NEXT:   addi.w    $t0, $t2, 72
# DIS32-NEXT:   srli.w    $t1, $t1, 2
# DIS32-NEXT:   ld.w      $t0, $t0, 4
# DIS32-NEXT:   jr        $t3

## 64-bit: .got.plt - .plt = 0x400d0 - 0x20020 = 4096*32+176
# DIS64-NEXT:   pcaddu12i $t2, 32
# DIS64-NEXT:   sub.d     $t1, $t1, $t3
# DIS64-NEXT:   ld.d      $t3, $t2, 176
# DIS64-NEXT:   addi.d    $t1, $t1, -44
# DIS64-NEXT:   addi.d    $t0, $t2, 176
# DIS64-NEXT:   srli.d    $t1, $t1, 1
# DIS64-NEXT:   ld.d      $t0, $t0, 8
# DIS64-NEXT:   jr        $t3

## 32-bit: &.got.plt[bar]-. = 0x40070-0x20040 = 4096*32+48
## 64-bit: &.got.plt[bar]-. = 0x400e0-0x20040 = 4096*32+160
# DIS:        20040: pcaddu12i $t3, 32
# DIS32-NEXT:        ld.w      $t3, $t3, 48
# DIS64-NEXT:        ld.d      $t3, $t3, 160
# DIS-NEXT:          jirl      $t1, $t3, 0
# DIS-NEXT:          nop

## 32-bit: &.got.plt[weak]-. = 0x40074-0x20050 = 4096*32+36
## 64-bit: &.got.plt[weak]-. = 0x400e8-0x20050 = 4096*32+152
# DIS:        20050: pcaddu12i $t3, 32
# DIS32-NEXT:        ld.w      $t3, $t3, 36
# DIS64-NEXT:        ld.d      $t3, $t3, 152
# DIS-NEXT:          jirl      $t1, $t3, 0
# DIS-NEXT:          nop

.global _start, foo, bar
.weak weak

_start:
    bl foo
    bl bar
    bl %plt(bar)
    bl weak

## foo is local and non-preemptible, no PLT is generated.
foo:
    ret
