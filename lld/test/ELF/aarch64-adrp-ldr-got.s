# REQUIRES: aarch64
# RUN: rm -rf %t && split-file %s %t

# RUN: llvm-mc -filetype=obj -triple=aarch64 %t/a.s -o %t/a.o
# RUN: llvm-mc -filetype=obj -triple=aarch64 %t/unpaired.s -o %t/unpaired.o
# RUN: llvm-mc -filetype=obj -triple=aarch64 %t/lone-ldr.s -o %t/lone-ldr.o
# RUN: llvm-mc -filetype=obj -triple=aarch64 %t/lone-adrp-ldr.s -o %t/lone-adrp-ldr.o
# RUN: llvm-mc -filetype=obj -triple=aarch64 %t/all-or-nothing.s -o %t/all-or-nothing.o
# RUN: llvm-mc -filetype=obj -triple=aarch64 %t/all-or-nothing-out-of-range.s -o %t/all-or-nothing-out-of-range.o

# RUN: ld.lld %t/a.o -T %t/out-of-adr-range.t -o %t/a
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

# RUN: ld.lld %t/a.o -T %t/within-adr-range.t -o %t/a
# RUN: llvm-objdump --no-show-raw-insn -d %t/a | FileCheck --check-prefix=ADR %s
# RUN: llvm-readelf -x .got %t/a | FileCheck --check-prefix=GOT-RELAX %s

## Symbol 'x' is nonpreemptible, the relaxation should be applied.
# ADR:        nop
# ADR-NEXT:   adr    x1

## Symbol 'x' does not have a GOT entry when relaxed.
# GOT-RELAX:      Hex dump of section '.got':
# GOT-RELAX-NEXT: 0x{{[0-9a-f]+}} 04100000 00000000 08100000 00000000
# GOT-RELAX-NEXT: 0x{{[0-9a-f]+}} 0c100000 00000000 10100000 00000000
# GOT-RELAX-NOT:  00100000

## Symbol 'x' is nonpreemptible, but --no-relax surpresses relaxations.
# RUN: ld.lld %t/a.o -T %t/out-of-adr-range.t --no-relax -o %t/no-relax
# RUN: llvm-objdump --no-show-raw-insn -d %t/no-relax | \
# RUN:   FileCheck --check-prefix=X1-NO-RELAX %s

# X1-NO-RELAX:      adrp   x1
# X1-NO-RELAX-NEXT: ldr

## Symbol 'x' is nonpreemptible, but the address is not within adrp range.
# RUN: ld.lld %t/a.o -T %t/out-of-range.t -o %t/out-of-range
# RUN: llvm-objdump --no-show-raw-insn -d %t/out-of-range | \
# RUN:   FileCheck --check-prefix=X1-NO-RELAX %s
# RUN: llvm-readelf -x .got %t/out-of-range | FileCheck --check-prefix=GOT-NO-RELAX %s

## Symbol 'x' has a GOT entry restored by relaxOnce when out of range.
# GOT-NO-RELAX:      Hex dump of section '.got':
# GOT-NO-RELAX-NEXT: 0x{{[0-9a-f]+}} 04100000 00000000 08100000 00000000
# GOT-NO-RELAX-NEXT: 0x{{[0-9a-f]+}} 0c100000 00000000 10100000 00000000
# GOT-NO-RELAX-NEXT: 0x{{[0-9a-f]+}} 00100000 00000000

## Symbol 'x' has upper bits set (e.g. HWASAN tag), relaxOnce restores the GOT entry
## even when output section VA is small.
# RUN: ld.lld %t/a.o -T %t/within-adr-range.t --defsym=x=0x4b00000000001000 -o %t/tagged
# RUN: llvm-objdump --no-show-raw-insn -d %t/tagged | \
# RUN:   FileCheck --check-prefix=X1-NO-RELAX %s
# RUN: llvm-readelf -x .got %t/tagged | FileCheck --check-prefix=GOT-TAGGED %s

# GOT-TAGGED:      Hex dump of section '.got':
# GOT-TAGGED-NEXT: 0x{{[0-9a-f]+}} 04100000 00000000 08100000 00000000
# GOT-TAGGED-NEXT: 0x{{[0-9a-f]+}} 0c100000 00000000 10100000 00000000
# GOT-TAGGED-NEXT: 0x{{[0-9a-f]+}} 00100000 0000004b

## Relocations do not appear in pairs, no relaxations should be applied for
## that symbol. We can still relax other symbols.
# RUN: ld.lld %t/unpaired.o -o %t/unpaired
# RUN: llvm-objdump --no-show-raw-insn -d %t/unpaired | \
# RUN:   FileCheck --check-prefix=UNPAIRED %s

# UNPAIRED:         adrp   x0
# UNPAIRED-NEXT:    b
# UNPAIRED-NEXT:    adrp   x0
# UNPAIRED:         ldr	   x0
## This is a different symbol.
# UNPAIRED:         nop
# UNPAIRED:         adr    x1

## Relocations do not appear in pairs, no relaxations should be applied.
# RUN: ld.lld %t/lone-ldr.o -o %t/lone-ldr
# RUN: llvm-objdump --no-show-raw-insn -d %t/lone-ldr | \
# RUN:   FileCheck --check-prefix=LONE-LDR %s

# LONE-LDR:         ldr	   x0

## A relaxable ADRP+LDR pair where no other GOT entries exist.
## When out of range, the GOT entry is restored in relaxOnce and .got must be
## retained even though it had 0 entries during removeUnusedSyntheticSections.
# RUN: ld.lld %t/lone-adrp-ldr.o -T %t/out-of-range.t -o %t/lone-out-of-range
# RUN: llvm-readelf -x .got %t/lone-out-of-range | FileCheck --check-prefix=LONE-OUT-OF-RANGE %s

# LONE-OUT-OF-RANGE:      Hex dump of section '.got':
# LONE-OUT-OF-RANGE-NEXT: 0x{{[0-9a-f]+}} 00100000 00000000

## Make sure that relaxation is not applied if not all adrp+ldr pairs for
## a given symbol can be relaxed. This is not legal, because there may be
## a branch destination between the adrp and ldr instructions. We can still
## perform the relaxation for other symbols, or the same symbol in a different
## section.
# RUN: ld.lld %t/all-or-nothing.o -o %t/all-or-nothing
# RUN: llvm-objdump --no-show-raw-insn -d %t/all-or-nothing | \
# RUN:   FileCheck --check-prefix=ALL-OR-NOTHING %s

# ALL-OR-NOTHING-LABEL: <_start>:
# ALL-OR-NOTHING: adrp   x1
# ALL-OR-NOTHING: ldr    x1
# ALL-OR-NOTHING: adrp   x1
# ALL-OR-NOTHING: ldr    x2
# ALL-OR-NOTHING: nop
# ALL-OR-NOTHING: adr    x1
# ALL-OR-NOTHING-LABEL: <foo>:
# ALL-OR-NOTHING: nop
# ALL-OR-NOTHING: adr    x1

## Make sure that relaxation is not applied if not all adrp+ldr pairs for
## a given symbol can be relaxed, even when the failure is due to one pair
## being out of range (>4GB) while another pair is in range (<=4GB).
# RUN: ld.lld %t/all-or-nothing-out-of-range.o -T %t/all-or-nothing-out-of-range.t -o %t/all-or-nothing-out-of-range
# RUN: llvm-objdump --no-show-raw-insn -d %t/all-or-nothing-out-of-range | \
# RUN:   FileCheck --check-prefix=ALL-OR-NOTHING-OUT-OF-RANGE %s
# RUN: llvm-readelf -x .got %t/all-or-nothing-out-of-range | FileCheck --check-prefix=ALL-OR-NOTHING-OUT-OF-RANGE-GOT %s

# ALL-OR-NOTHING-OUT-OF-RANGE-LABEL: <_start>:
# ALL-OR-NOTHING-OUT-OF-RANGE-NEXT:  adrp   x1
# ALL-OR-NOTHING-OUT-OF-RANGE-NEXT:  ldr    x1
# ALL-OR-NOTHING-OUT-OF-RANGE-NEXT:  b      0x{{[0-9a-f]+}} <_start+0x2010>
# ALL-OR-NOTHING-OUT-OF-RANGE:       adrp   x1
# ALL-OR-NOTHING-OUT-OF-RANGE-NEXT:  ldr    x1
# ALL-OR-NOTHING-OUT-OF-RANGE-NOT:   add    x1, x1

# ALL-OR-NOTHING-OUT-OF-RANGE-GOT:      Hex dump of section '.got':
# ALL-OR-NOTHING-OUT-OF-RANGE-GOT-NEXT: 0x{{[0-9a-f]+}} 00100000 00000000

## This linker script ensures that .rodata and .text are sufficiently (>1M)
## far apart so that the adrp + ldr pair cannot be relaxed to adr + nop.
#--- out-of-adr-range.t
SECTIONS {
 .rodata 0x1000: { *(.rodata) }
 .text   0x200100: { *(.text) }
}

## This linker script ensures that .rodata and .text are sufficiently (<1M)
## close to each other so that the adrp + ldr pair can be relaxed to nop + adr.
#--- within-adr-range.t
SECTIONS {
 .rodata 0x1000: { *(.rodata) }
 .text   0x2000: { *(.text) }
}

## This linker script ensures that .rodata and .text are sufficiently (>4GB)
## far apart so that the adrp + ldr pair cannot be relaxed.
#--- out-of-range.t
SECTIONS {
 .rodata 0x1000: { *(.rodata) }
 .text   0x100002000: { *(.text) }
}

## This linker script ensures that the first pair in .text is within 4GB
## of .rodata while the second pair in the same section is out of range (>4GB).
#--- all-or-nothing-out-of-range.t
SECTIONS {
 .rodata 0x1000: { *(.rodata) }
 .text   0x100000000: { *(.text) }
}

#--- a.s
.rodata
.globl x
.hidden x
x:
.word 10
.hidden y
y:
.word 10
.hidden z
z:
.word 10
.hidden u
u:
.word 10
.hidden v
v:
.word 10
.text
.global _start
_start:
  adrp    x1, :got:x
  ldr     x1, [x1, #:got_lo12:x]
  adrp    x2, :got:y+1
  ldr     x2, [x2, #:got_lo12:y]
  adrp    x3, :got:z
  ldr     x3, [x3, #:got_lo12:z+8]
  adrp    x4, :got:u
  ldr     x5, [x4, #:got_lo12:u]
  adrp    x6, :got:v
  ldr     x6, [x0, #:got_lo12:v]

#--- unpaired.s
.text
.globl x
.hidden x
x:
  nop
.hidden y
y:
  nop
.global _start
_start:
  adrp    x0, :got:x
  b L
  adrp    x0, :got:x
L:
  ldr     x0, [x0, #:got_lo12:x]
  adrp    x1, :got:y
  ldr     x1, [x1, #:got_lo12:y]

#--- lone-ldr.s
.text
.globl x
.hidden x
x:
  nop
.global _start
_start:
  ldr     x0, [x0, #:got_lo12:x]

#--- all-or-nothing.s
.rodata
.globl x
.hidden x
x:
.word 10
.hidden y
y:
.word 10
.text
.global _start
_start:
  adrp    x1, :got:x
  ldr     x1, [x1, #:got_lo12:x]
  adrp    x1, :got:x
  ldr     x2, [x1, #:got_lo12:x]
  adrp    x1, :got:y
  ldr     x1, [x1, #:got_lo12:y]

.section .text.foo
.global foo
foo:
  adrp    x1, :got:x
  ldr     x1, [x1, #:got_lo12:x]

#--- lone-adrp-ldr.s
.rodata
.globl x
.hidden x
x:
.word 10
.text
.global _start
_start:
  adrp    x1, :got:x
  ldr     x1, [x1, #:got_lo12:x]

#--- all-or-nothing-out-of-range.s
.rodata
.globl x
.hidden x
x:
.word 10
.text
.global _start
_start:
  adrp    x1, :got:x
  ldr     x1, [x1, #:got_lo12:x]
  b       .L1
  .space  0x2000
  adrp    x1, :got:x
.L1:
  ldr     x1, [x1, #:got_lo12:x]
