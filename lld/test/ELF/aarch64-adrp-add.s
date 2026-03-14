# REQUIRES: aarch64
# RUN: rm -rf %t && split-file %s %t

# RUN: llvm-mc -filetype=obj -triple=aarch64 %t/a.s -o %t/a.o
# RUN: ld.lld %t/a.o -T %t/out-of-adr-range-low.t -o %t/a-low
# RUN: llvm-objdump --no-show-raw-insn -d %t/a-low | FileCheck %s --check-prefix=OUT-OF-RANGE
# RUN: ld.lld %t/a.o -T %t/out-of-adr-range-high.t -o %t/a-high
# RUN: llvm-objdump --no-show-raw-insn -d %t/a-high | FileCheck %s --check-prefix=OUT-OF-RANGE

# OUT-OF-RANGE:      adrp  x30
# OUT-OF-RANGE-NEXT: add   x30, x30

# RUN: llvm-mc -filetype=obj -triple=aarch64 %t/a.s -o %t/a.o
# RUN: ld.lld %t/a.o -T %t/within-adr-range-low.t -o %t/a-low
# RUN: llvm-objdump --no-show-raw-insn -d %t/a-low | FileCheck %s --check-prefix=IN-RANGE-LOW

# IN-RANGE-LOW:      nop
# IN-RANGE-LOW-NEXT: adr   x30
# IN-RANGE-LOW-NEXT: adrp  x1
# IN-RANGE-LOW-NEXT: add   x1
# IN-RANGE-LOW-NEXT: adrp  x15
# IN-RANGE-LOW-NEXT: add   x15

## ADRP and ADD use different registers, no relaxations should be applied.
# IN-RANGE-LOW-NEXT: adrp  x2
# IN-RANGE-LOW-NEXT: add   x3, x2

## ADRP and ADD use different registers, no relaxations should be applied.
# IN-RANGE-LOW-NEXT: adrp  x2
# IN-RANGE-LOW-NEXT: add   x2, x3

# RUN: ld.lld %t/a.o -T %t/within-adr-range-high.t -o %t/a-high
# RUN: llvm-objdump --no-show-raw-insn -d %t/a-high | FileCheck %s --check-prefix=IN-RANGE-HIGH

# IN-RANGE-HIGH:      nop
# IN-RANGE-HIGH-NEXT: adr   x30
# IN-RANGE-HIGH-NEXT: nop
# IN-RANGE-HIGH-NEXT: adr   x1
# IN-RANGE-HIGH-NEXT: nop
# IN-RANGE-HIGH-NEXT: adr   x15

## ADRP and ADD use different registers, no relaxations should be applied.
# IN-RANGE-HIGH-NEXT: adrp  x2
# IN-RANGE-HIGH-NEXT: add   x3, x2

## ADRP and ADD use different registers, no relaxations should be applied.
# IN-RANGE-HIGH-NEXT: adrp  x2
# IN-RANGE-HIGH-NEXT: add   x2, x3

# RUN: llvm-mc -filetype=obj -triple=aarch64 %t/a.s -o %t/a.o
# RUN: ld.lld %t/a.o -T %t/within-adr-range-low.t --no-relax -o %t/a
## --no-relax disables relaxations.
# RUN: llvm-objdump --no-show-raw-insn -d %t/a | FileCheck %s --check-prefix=OUT-OF-RANGE

## .rodata and .text are close to each other,
## the adrp + add pair can be relaxed to nop + adr, moreover, the address difference
## is equal to the lowest allowed value.
#--- within-adr-range-low.t
SECTIONS {
 .rodata 0x1000: { *(.rodata) }
 .text   0x100ffc: { *(.text) }
}

## .rodata and .text are far apart,
## the adrp + add pair cannot be relaxed to nop + adr, moreover, the address difference
## is equal to the lowest allowed value minus one.
#--- out-of-adr-range-low.t
SECTIONS {
 .rodata 0x1000: { *(.rodata) }
 .text   0x100ffd: { *(.text) }
}

## .rodata and .text are close to each other,
## the adrp + add pair can be relaxed to nop + adr, moreover, the address difference
## is equal to the highest allowed value.
#--- within-adr-range-high.t
SECTIONS {
 .text   0x1000: { *(.text) }
 .rodata 0x101003: { *(.rodata) }
}

## .rodata and .text are far apart,
## the adrp + add pair cannot be relaxed to nop + adr, moreover, the address difference
## is equal to the highest allowed value plus one.
#--- out-of-adr-range-high.t
SECTIONS {
 .text   0x1000: { *(.text) }
 .rodata 0x101004: { *(.rodata) }
}

#--- a.s
.rodata
x:
.word 10
.text
.global _start
_start:
  adrp    x30, x
  add     x30, x30, :lo12:x
  adrp    x1, x
  add     x1, x1, :lo12:x
  adrp    x15, x
  add     x15, x15, :lo12:x
  adrp    x2, x
  add     x3, x2, :lo12:x
  adrp    x2, x
  add     x2, x3, :lo12:x
