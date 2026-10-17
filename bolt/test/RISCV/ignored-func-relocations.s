## Calls and address materialization in ignored code contain specifier
## expressions. Keep their targets in place until instruction relocations
## can be rewritten safely, including paired high/low relocations.
## Check positive and negative addends and continue scanning after each pair.

# RUN: llvm-mc -triple=riscv64 -filetype=obj %s -o %t.o
# RUN: ld.lld --emit-relocs --no-relax %t.o -o %t.exe
# RUN: llvm-bolt %t.exe --skip-funcs=source -o %t.bolt
# RUN: llvm-nm --defined-only -n %t.exe > %t.nm
# RUN: llvm-nm --defined-only -n %t.bolt >> %t.nm
# RUN: FileCheck %s --input-file=%t.nm

# CHECK: [[SOURCE:[0-9a-f]+]] T source
# CHECK: [[CALLEE:[0-9a-f]+]] T callee
# CHECK: [[ALTERNATE:[0-9a-f]+]] T alternate
# CHECK: [[SOURCE]] T source
# CHECK: [[CALLEE]] T callee
# CHECK: [[ALTERNATE]] T alternate

  .option norvc
  .option norelax
  .text
  .globl _start
  .type _start, @function
_start:
  call source
  li a7, 93
  ecall
  .size _start, .-_start

  .globl source
  .type source, @function
source:
  addi sp, sp, -16
  sd ra, 0(sp)
  call callee
  bnez a0, .Lfail
  call t0, alternate
  lla a1, callee+4
  lla a2, expected
  ld a3, 0(a2)
  bne a1, a3, .Lfail
  lla a1, callee-4
  ld a3, 8(a2)
  bne a1, a3, .Lfail
  lui a1, %hi(callee+8)
  addi a1, a1, %lo(callee+8)
  ld a3, 16(a2)
  bne a1, a3, .Lfail
  li a0, 0
  j .Lreturn
.Lfail:
  li a0, 1
.Lreturn:
  ld ra, 0(sp)
  addi sp, sp, 16
  ret
  .size source, .-source

  .globl callee
  .type callee, @function
callee:
  li a0, 1
  li a0, 0
  .rept 7
  nop
  .endr
  ret
  .size callee, .-callee

  .globl alternate
  .type alternate, @function
alternate:
  .rept 8
  nop
  .endr
  jr t0
  .size alternate, .-alternate

  .data
  .p2align 3
expected:
  .quad callee+4
  .quad callee-4
  .quad callee+8
