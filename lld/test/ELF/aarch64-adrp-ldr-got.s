# REQUIRES: aarch64
# RUN: split-file %s %t

# RUN: llvm-mc -filetype=obj -triple=aarch64 %t/a.s -o %t/a.o
# RUN: llvm-mc -filetype=obj -triple=aarch64 %t/unpaired.s -o %t/unpaired.o
# RUN: llvm-mc -filetype=obj -triple=aarch64 %t/lone-ldr.s -o %t/lone-ldr.o

# RUN: ld.lld %t/a.o -T %t/linker.t -o %t/a
# RUN: llvm-objdump --no-show-raw-insn -d %t/a | FileCheck %s

## Symbol 'x' is nonpreemptible, the relaxation should be applied.
## This test verifies the encoding when the register x1 is used.
# CHECK:      adrp   x1
# CHECK-NEXT: add    x1, x1

## ADRP contains a nonzero addend, no relaxations should be applied.
# CHECK-NEXT: adrp   x2
# CHECK-NEXT: ldr

## LDR contains a nonzero addend, no relaxations should be applied.
# CHECK-NEXT: adrp   x3
# CHECK-NEXT: ldr

## LDR and ADRP use different registers, no relaxations should be applied.
# CHECK-NEXT: adrp   x4
# CHECK-NEXT: ldr

## LDR and ADRP use different registers, no relaxations should be applied.
# CHECK-NEXT: adrp   x6
# CHECK-NEXT: ldr

## Symbol 'x' is nonpreemptible, but --no-relax surpresses relaxations.
# RUN: ld.lld %t/a.o -T %t/linker.t --no-relax -o %t/no-relax
# RUN: llvm-objdump --no-show-raw-insn -d %t/no-relax | \
#   FileCheck --check-prefix=X1-NO-RELAX %s

# X1-NO-RELAX:      adrp   x1
# X1-NO-RELAX-NEXT: ldr

## Symbol 'x' is nonpreemptible, but the address is not within adrp range.
# RUN: ld.lld %t/a.o -T %t/out-of-range.t -o %t/out-of-range
# RUN: llvm-objdump --no-show-raw-insn -d %t/out-of-range | \
# RUN:   FileCheck --check-prefix=X1-NO-RELAX %s

## Relocations do not appear in pairs, no relaxations should be applied.
# RUN: ld.lld %t/unpaired.o -o %t/unpaired
# RUN: llvm-objdump --no-show-raw-insn -d %t/unpaired | \
# RUN:   FileCheck --check-prefix=UNPAIRED %s

# UNPAIRED:         adrp   x0
# UNPAIRED-NEXT:    b
# UNPAIRED-NEXT:    adrp   x0
# UNPAIRED:         ldr	   x0

## Relocations do not appear in pairs, no relaxations should be applied.
# RUN: ld.lld %t/lone-ldr.o -o %t/lone-ldr
# RUN: llvm-objdump --no-show-raw-insn -d %t/lone-ldr | \
# RUN:   FileCheck --check-prefix=LONE-LDR %s

# LONE-LDR:         ldr	   x0

## This linker script ensures that .rodata and .text are sufficiently (>1M)
## far apart so that the adrp + ldr pair cannot be relaxed to adr + nop.
#--- linker.t
SECTIONS {
 .rodata 0x1000: { *(.rodata) }
 .text   0x200100: { *(.text) }
}

## This linker script ensures that .rodata and .text are sufficiently (>4GB)
## far apart so that the adrp + ldr pair cannot be relaxed.
#--- out-of-range.t
SECTIONS {
 .rodata 0x1000: { *(.rodata) }
 .text   0x100002000: { *(.text) }
}

#--- a.s
.rodata
.hidden x
x:
.word 10
.text
.global _start
_start:
  adrp    x1, :got:x
  ldr     x1, [x1, #:got_lo12:x]
  adrp    x2, :got:x+1
  ldr     x2, [x2, #:got_lo12:x]
  adrp    x3, :got:x
  ldr     x3, [x3, #:got_lo12:x+8]
  adrp    x4, :got:x
  ldr     x5, [x4, #:got_lo12:x]
  adrp    x6, :got:x
  ldr     x6, [x0, #:got_lo12:x]

#--- unpaired.s
.text
.hidden x
x:
  nop
.global _start
_start:
  adrp    x0, :got:x
  b L
  adrp    x0, :got:x
L:
  ldr     x0, [x0, #:got_lo12:x]

#--- lone-ldr.s
.text
.hidden x
x:
  nop
.global _start
_start:
  ldr     x0, [x0, #:got_lo12:x]
