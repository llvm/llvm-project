## An ignored caller may use t0 as its link register. An entry trampoline
## would overwrite the return address, so keep the callee at its old address.
## The callee is large enough to accommodate an entry patch.

# RUN: llvm-mc -triple=riscv64 -filetype=obj %s -o %t.o
# RUN: ld.lld --emit-relocs %t.o -o %t.exe
# RUN: llvm-bolt %t.exe --skip-funcs=source -o %t.bolt
# RUN: llvm-nm --defined-only -n %t.exe > %t.nm
# RUN: llvm-nm --defined-only -n %t.bolt >> %t.nm
# RUN: FileCheck %s --input-file=%t.nm

# CHECK: [[HELPER:[0-9a-f]+]] T helper
# CHECK: [[SOURCE:[0-9a-f]+]] T source
# CHECK: [[TAIL:[0-9a-f]+]] T tail
# CHECK: [[HELPER]] T helper
# CHECK: [[SOURCE]] T source
# CHECK: [[TAIL]] T tail

  .option norvc
  .text
  .globl _start
  .type _start, @function
_start:
  li a0, 0
  jal ra, source
  li a7, 93
  ecall
  .size _start, .-_start

## Exercise a target that has already been disassembled when source is scanned.
  .globl helper
  .type helper, @function
helper:
  .rept 8
  nop
  .endr
  jr t0
  .size helper, .-helper

  .globl source
  .type source, @function
source:
  jal t0, helper
  li t0, 17
  beq a0, zero, tail
  li a0, 1
  ret
  .size source, .-source

## t0 can also be live across an ordinary conditional branch. This target has
## not yet been disassembled when source is scanned.
  .globl tail
  .type tail, @function
tail:
  li t1, 17
  sub a0, t0, t1
  sltu a0, zero, a0
  .rept 6
  nop
  .endr
  ret
  .size tail, .-tail

  .reloc 0, R_RISCV_NONE
