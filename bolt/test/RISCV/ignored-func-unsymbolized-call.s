## Scanning retained AUIPC/JALR pairs must work without relocations or new
## entry symbols. The odd immediate targets the second instruction in target.

# RUN: llvm-mc -triple=riscv64 -filetype=obj %s -o %t.o
# RUN: ld.lld --emit-relocs --no-relax %t.o -o %t.exe
# RUN: llvm-bolt %t.exe --skip-funcs=source -o %t.bolt
# RUN: llvm-nm --defined-only -n %t.exe > %t.nm
# RUN: llvm-nm --defined-only -n %t.bolt >> %t.nm
# RUN: FileCheck %s --input-file=%t.nm --implicit-check-not=__ENTRY_
# RUN: llvm-mc -triple=riscv32 -filetype=obj %s -o %t.32.o
# RUN: ld.lld --emit-relocs --no-relax %t.32.o -o %t.32.exe
# RUN: llvm-bolt %t.32.exe --skip-funcs=source -o %t.32.bolt
# RUN: llvm-nm --defined-only -n %t.32.exe > %t.32.nm
# RUN: llvm-nm --defined-only -n %t.32.bolt >> %t.32.nm
# RUN: FileCheck %s --input-file=%t.32.nm --implicit-check-not=__ENTRY_

# CHECK: [[SOURCE:[0-9a-f]+]] T source
# CHECK: [[TARGET:[0-9a-f]+]] T target
# CHECK: [[SOURCE]] T source
# CHECK: [[TARGET]] T target

  .option norvc
  .option norelax
  .text
  .globl _start
  .type _start, @function
_start:
  call source
  lla a1, target+4
  sub a0, a0, a1
  snez a0, a0
  li a7, 93
  ecall
  .size _start, .-_start

  .globl source
  .type source, @function
source:
  auipc t0, 0
  jalr t0, 17(t0)
  ret
  .size source, .-source

  .globl target
  .type target, @function
target:
  li a0, 1
  lla a0, target+4
  .rept 8
  nop
  .endr
  jr t0
  .size target, .-target
