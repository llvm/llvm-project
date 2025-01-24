## This test checks that disassemble stage works properly
## JT with indirect branch
## 1) nop + adr pair instructions
## 2) sub + ldr pair instructions
## 3) adrp + ldr pair instructions

# REQUIRES: system-linux

# RUN: rm -rf %t && split-file %s %t

## Prepare binary (1)
# RUN: llvm-mc -filetype=obj -triple aarch64-unknown-unknown %t/jt_nop_adr.s \
# RUN:    -o %t/jt_nop_adr.o
# RUN: %clang %cflags --target=aarch64-unknown-linux %t/jt_nop_adr.o \
# RUN:  -Wl,-q -Wl,-z,now, -Wl,-T,%t/within-adr-range.t -o %t/jt_nop_adr.exe
# RUN: llvm-objdump --no-show-raw-insn -d %t/jt_nop_adr.exe | FileCheck \
# RUN:    --check-prefix=JT-RELAXED %s

# JT-RELAXED: <_start>:
# JT-RELAXED-NEXT:  nop
# JT-RELAXED-NEXT:  adr {{.*}}x3

# RUN: llvm-bolt %t/jt_nop_adr.exe -o %t/jt_nop_adr.bolt -v 2 2>&1 | FileCheck %s
# CHECK-NOT: Failed to match

## This linker script ensures that .rodata and .text are sufficiently (<1M)
## close to each other so that the adrp + ldr pair can be relaxed to nop + adr.
#--- within-adr-range.t
SECTIONS {
 .rodata 0x1000: { *(.rodata) }
 .text   0x2000: { *(.text) }
 .rela.rodata :    { *(.rela.rodata) }
}

## Prepare binary (2)
# RUN: llvm-mc -filetype=obj -triple aarch64-unknown-unknown %t/jt_sub_ldr.s \
# RUN:   -o %t/jt_sub_ldr.o
# RUN: %clang %cflags --target=aarch64-unknown-linux %t/jt_sub_ldr.o \
# RUN:  -Wl,-q -Wl,-z,now -o %t/jt_sub_ldr.exe
# RUN: llvm-objdump --no-show-raw-insn -d %t/jt_sub_ldr.exe | FileCheck \
# RUN:    --check-prefix=JT-SUB-LDR %s

# JT-SUB-LDR: <_start>:
# JT-SUB-LDR-NEXT:  sub
# JT-SUB-LDR-NEXT:  ldr

# RUN: llvm-bolt %t/jt_sub_ldr.exe -o %t/jt_sub_ldr.bolt -v 2 2>&1 | FileCheck \
# RUN:    --check-prefix=JT-BOLT-SUBLDR %s
# JT-BOLT-SUBLDR-NOT: Failed to match

## Prepare binary (3)
# RUN: llvm-mc -filetype=obj -triple aarch64-unknown-unknown %t/jt_adrp_ldr.s \
# RUN:    -o %t/jt_adrp_ldr.o
# RUN: %clang %cflags --target=aarch64-unknown-linux %t/jt_adrp_ldr.o \
# RUN:  -Wl,-q -Wl,-z,now  -Wl,--no-relax -o %t/jt_adrp_ldr.exe
# RUN: llvm-objdump --no-show-raw-insn -d %t/jt_adrp_ldr.exe | FileCheck \
# RUN:   --check-prefix=JT-ADRP-LDR %s

# JT-ADRP-LDR: <_start>:
# JT-ADRP-LDR-NEXT:  adrp
# JT-ADRP-LDR-NEXT:  ldr

# RUN: llvm-bolt %t/jt_adrp_ldr.exe -o %t/jt_adrp_ldr.bolt -v 2 2>&1 | FileCheck \
# RUN:   --check-prefix=JT-BOLT-ADRP-LDR %s
# JT-BOLT-ADRP-LDR-NOT: Failed to match

#--- jt_nop_adr.s
  .globl _start
  .type  _start, %function
_start:
  adrp    x3, :got:jump_table
  ldr     x3, [x3, #:got_lo12:jump_table]
  ldrh    w3, [x3, x1, lsl #1]
  adr     x1, test2_0
  add     x3, x1, w3, sxth #2
  br      x3
test2_0:
  ret
test2_1:
  ret

  .section .rodata,"a",@progbits
jump_table:
  .hword  (test2_0-test2_0)>>2
  .hword  (test2_1-test2_0)>>2


#--- jt_sub_ldr.s
  .globl _start
  .type  _start, %function
_start:
  sub     x1, x29, #0x4, lsl #12
  ldr     x1, [x1, #14352]
  ldrh    w1, [x1, w3, uxtw #1]
  adr     x3, test2_0
  add     x1, x3, w1, sxth #2
  br      x1
test2_0:
  ret
test2_1:
  ret

  .section .rodata,"a",@progbits
jump_table:
  .hword  (test2_0-test2_0)>>2
  .hword  (test2_1-test2_0)>>2


#--- jt_adrp_ldr.s
  .globl _start
  .type  _start, %function
_start:
  adrp    x3, :got:jump_table
  ldr     x3, [x3, #:got_lo12:jump_table]
  ldrh    w3, [x3, x1, lsl #1]
  adr     x1, test2_0
  add     x3, x1, w3, sxth #2
  br      x3
test2_0:
  ret
test2_1:
  ret

  .section .rodata,"a",@progbits
jump_table:
  .hword  (test2_0-test2_0)>>2
  .hword  (test2_1-test2_0)>>2
