# REQUIRES: system-linux

# RUN: %clang %cflags64 -march=rv64gc %s -o %t.exe
# RUN: link_fdata --no-lbr --nmtool llvm-nm %s %t.exe %t.fdata
# RUN: llvm-bolt %t.exe -relocs -o %t.out -data %t.fdata \
# RUN:   -frame-opt=all -simplify-conditional-tail-calls=false \
# RUN:   -eliminate-unreachable=false
# RUN: llvm-objdump -d --no-show-raw-insn %t.out | \
# RUN:   FileCheck %s

  .text
  .reloc 0, R_RISCV_NONE
  .globl _start
  .type _start, @function
_start:
  .cfi_startproc
# FDATA: 1 _start #_start# 100
  addi sp, sp, -0x1f0
  .cfi_adjust_cfa_offset 0x1f0
  sd ra, 0x1e8(sp)
  .cfi_offset ra, -8
  addi sp, sp, -0x800
  .cfi_adjust_cfa_offset 0x800
  addi sp, sp, -0x6f0
  .cfi_adjust_cfa_offset 0x6f0
  sd a0, 0(sp)
  call callee
  ld a0, 0(sp)
  addi sp, sp, 0x7f0
  .cfi_adjust_cfa_offset -0x7f0
  addi sp, sp, 0x700
  .cfi_adjust_cfa_offset -0x700
  ld ra, 0x1e8(sp)
  .cfi_restore ra
  addi sp, sp, 0x1f0
  .cfi_adjust_cfa_offset -0x1f0
  ret
  .cfi_endproc
  .size _start, .-_start

  .type callee, @function
callee:
  ret
  .size callee, .-callee

# CHECK-LABEL: <_start>:
# CHECK:       addi sp, sp, -0x1f0
# CHECK:       addi sp, sp, -0x800
# CHECK-NEXT:  addi sp, sp, -0x6f0
# CHECK:       addi sp, sp, 0x7f0
# CHECK-NEXT:  addi sp, sp, 0x700
# CHECK:       ld ra, 0x1e8(sp)
# CHECK:       addi sp, sp, 0x1f0
