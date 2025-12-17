# REQUIRES: loongarch
# RUN: rm -rf %t && split-file %s %t && cd %t

# RUN: llvm-mc --filetype=obj --triple=loongarch64 a.s -o a.o
# RUN: llvm-mc --filetype=obj --triple=loongarch64 unpaired.s -o unpaired.o
# RUN: llvm-mc --filetype=obj --triple=loongarch64 lone-ldr.s -o lone-ldr.o

# RUN: ld.lld a.o -T within-range.t -o a
# RUN: llvm-objdump -d --no-show-raw-insn a | FileCheck %s

## This test verifies the encoding when the register $a0 is used.
# CHECK:      pcalau12i $a0, 0
# CHECK-NEXT: addi.d    $a0, $a0, -2048

## PCALAU12I contains a nonzero addend, no relaxations should be applied.
# CHECK-NEXT: pcalau12i $a1, 2
# CHECK-NEXT: ld.d      $a1, $a1, -2048

## LD contains a nonzero addend, no relaxations should be applied.
# CHECK-NEXT: pcalau12i $a2, 2
# CHECK-NEXT: ld.d      $a2, $a2, -2040

## PCALAU12I and LD use different registers, no relaxations should be applied.
# CHECK-NEXT: pcalau12i $a3, 2
# CHECK-NEXT: ld.d      $a4, $a3, -2048

## PCALAU12I and LD use different registers, no relaxations should be applied.
# CHECK-NEXT: pcalau12i $a5, 2
# CHECK-NEXT: ld.d      $a5, $a6, -2048

# RUN: ld.lld a.o -T underflow-range.t -o a-underflow
# RUN: llvm-objdump -d --no-show-raw-insn a-underflow | FileCheck --check-prefix=OUTRANGE %s

# RUN: ld.lld a.o -T overflow-range.t -o a-overflow
# RUN: llvm-objdump -d --no-show-raw-insn a-overflow | FileCheck --check-prefix=OUTRANGE %s

# OUTRANGE:      pcalau12i $a0, 1
# OUTRANGE-NEXT: ld.d      $a0, $a0, 0

## Relocations do not appear in pairs, no relaxations should be applied.
# RUN: ld.lld unpaired.o -T within-range.t  -o unpaired
# RUN: llvm-objdump --no-show-raw-insn -d unpaired | FileCheck --check-prefix=UNPAIRED %s

# UNPAIRED:         pcalau12i $a0, 2
# UNPAIRED-NEXT:    b         8
# UNPAIRED-NEXT:    pcalau12i $a0, 2
# UNPAIRED:         ld.d      $a0, $a0, -2048

## Relocations do not appear in pairs, no relaxations should be applied.
# RUN: ld.lld lone-ldr.o -T within-range.t -o lone-ldr
# RUN: llvm-objdump --no-show-raw-insn -d lone-ldr | FileCheck --check-prefix=LONE-LDR %s

# LONE-LDR:         ld.d   $a0, $a0, -2048

## 32-bit code is mostly the same. We only test a few variants.
# RUN: llvm-mc --filetype=obj --triple=loongarch32 a.32.s -o a.32.o
# RUN: ld.lld a.32.o -T within-range.t -o a32
# RUN: llvm-objdump -d --no-show-raw-insn a32 | FileCheck --check-prefix=CHECK32 %s

## This test verifies the encoding when the register $a0 is used.
# CHECK32:      pcalau12i $a0, 0
# CHECK32-NEXT: addi.w    $a0, $a0, -2048


## This linker script ensures that .rodata and .text are sufficiently close to
## each other so that the pcalau12i + ld pair can be relaxed to pcalau12i + add.
#--- within-range.t
SECTIONS {
 .rodata 0x1800: { *(.rodata) }
 .text   0x2800: { *(.text) }
 .got    0x3800: { *(.got) }
}

## This linker script ensures that .rodata and .text are sufficiently far apart
## so that the pcalau12i + ld pair cannot be relaxed to pcalau12i + add.
#--- underflow-range.t
SECTIONS {
 .rodata 0x800-4: { *(.rodata) }
 .got    0x80002000: { *(.got) }
 .text   0x80001000: { *(.text) }  /* (0x800-4)+2GB+0x800+4 */
}

#--- overflow-range.t
SECTIONS {
 .text   0x1000: { *(.text) }
 .got    0x2000: { *(.got) }
 .rodata 0x80000800 : { *(.rodata) }  /* 0x1000+2GB-0x800 */
}

#--- a.s
## Symbol 'x' is nonpreemptible, the optimization should be applied.
.rodata
.hidden x
x:
.word 10

.text
.global _start
_start:
  pcalau12i $a0, %got_pc_hi20(x)
  ld.d      $a0, $a0, %got_pc_lo12(x)
  pcalau12i $a1, %got_pc_hi20(x+1)
  ld.d      $a1, $a1, %got_pc_lo12(x)
  pcalau12i $a2, %got_pc_hi20(x)
  ld.d      $a2, $a2, %got_pc_lo12(x+8)
  pcalau12i $a3, %got_pc_hi20(x)
  ld.d      $a4, $a3, %got_pc_lo12(x)
  pcalau12i $a5, %got_pc_hi20(x)
  ld.d      $a5, $a6, %got_pc_lo12(x)

#--- unpaired.s
.text
.hidden x
x:
  nop
.global _start
_start:
  pcalau12i $a0, %got_pc_hi20(x)
  b L
  pcalau12i $a0, %got_pc_hi20(x)
L:
  ld.d      $a0, $a0, %got_pc_lo12(x)

#--- lone-ldr.s
.text
.hidden x
x:
  nop
.global _start
_start:
  ld.d     $a0, $a0, %got_pc_lo12(x)


#--- a.32.s
## Symbol 'x' is nonpreemptible, the optimization should be applied.
.rodata
.hidden x
x:
.word 10

.text
.global _start
_start:
  pcalau12i $a0, %got_pc_hi20(x)
  ld.w      $a0, $a0, %got_pc_lo12(x)
