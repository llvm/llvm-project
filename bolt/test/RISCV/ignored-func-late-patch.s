## Rejecting a short caller during entry patching also pins its callee.
## Discard the callee's pending patch even when it was checked first.

# RUN: llvm-mc -triple=riscv64 -filetype=obj %s -o %t.o
# RUN: ld.lld --emit-relocs --no-relax %t.o -o %t.exe
# RUN: llvm-bolt %t.exe --force-patch -o %t.bolt 2>&1 | FileCheck %s --check-prefix=DIAG
# RUN: llvm-nm --defined-only -n %t.exe > %t.nm
# RUN: llvm-nm --defined-only -n %t.bolt >> %t.nm
# RUN: FileCheck %s --input-file=%t.nm --check-prefix=ADDR --implicit-check-not=target.org.0
# RUN: llvm-objdump -d --section=.bolt.org.text %t.bolt | FileCheck %s --check-prefix=CODE
# RUN: llvm-mc -triple=riscv32 -filetype=obj %s -o %t.32.o
# RUN: ld.lld --emit-relocs --no-relax %t.32.o -o %t.32.exe
# RUN: llvm-bolt %t.32.exe --force-patch -o %t.32.bolt 2>&1 | FileCheck %s --check-prefix=DIAG
# RUN: llvm-nm --defined-only -n %t.32.exe > %t.32.nm
# RUN: llvm-nm --defined-only -n %t.32.bolt >> %t.32.nm
# RUN: FileCheck %s --input-file=%t.32.nm --check-prefix=ADDR --implicit-check-not=target.org.0
# RUN: llvm-objdump -d --section=.bolt.org.text %t.32.bolt | FileCheck %s --check-prefix=CODE

# DIAG: BOLT-WARNING: failed to patch entries in source
# ADDR: [[TARGET:[0-9a-f]+]] T target
# ADDR: [[SOURCE:[0-9a-f]+]] T source
# ADDR: [[TARGET]] T target
# ADDR: [[SOURCE]] T source
# CODE-LABEL: <target>:
# CODE-NEXT: li a0, {{(0x)?0}}
# CODE-NEXT: nop

  .option norvc
  .option norelax
  .text
  .globl target
  .type target, @function
target:
  li a0, 0
  .rept 40
  nop
  .endr
  jr t0
  .size target, .-target

  .globl source
  .type source, @function
source:
  jal t0, target
  ret
  .size source, .-source

  .globl _start
  .type _start, @function
_start:
  call source
  li a7, 93
  ecall
  .size _start, .-_start
