## Reject a data-island target again when ignoring the source rescans its
## references. The data word is an instruction encoding, as in assembly that
## emits instructions as data; its mapping symbol still identifies data.
## Both the source and target must retain their original addresses and bytes.

# RUN: llvm-mc -triple=riscv64 -filetype=obj %s -o %t.o
# RUN: ld.lld --emit-relocs -Ttext=0x10000 %t.o -o %t.exe
# RUN: llvm-objcopy --dump-section=.text=%t.original.text %t.exe
# RUN: llvm-bolt %t.exe -o %t.bolt 2>&1 | FileCheck %s --check-prefix=DIAG
# RUN: llvm-nm --defined-only %t.bolt | FileCheck %s --check-prefix=ADDR --implicit-check-not=__ENTRY_
# RUN: llvm-objcopy --dump-section=.bolt.org.text=%t.output.text %t.bolt
# RUN: cmp %t.original.text %t.output.text
# RUN: llvm-bolt %t.exe -o %t.skip.bolt --skip-funcs=_start 2>&1 | FileCheck %s --check-prefix=DIAG
# RUN: llvm-nm --defined-only %t.skip.bolt | FileCheck %s --check-prefix=ADDR --implicit-check-not=__ENTRY_
# RUN: llvm-objcopy --dump-section=.bolt.org.text=%t.skip.text %t.skip.bolt
# RUN: cmp %t.original.text %t.skip.text

## The same invalid reference must not be diagnosed repeatedly during rescans.
# DIAG: BOLT-WARNING: ignoring entry point at address 0x1000c in constant island of function target
# DIAG-NOT: ignoring entry point
# DIAG-NOT: BOLT-ERROR
# ADDR-DAG: 0000000000010000 T _start
# ADDR-DAG: 0000000000010008 T target

  .text
  .option norvc
  .globl _start
  .type _start,@function
_start:
  beq t0, t1, .Ldata
  ret
  .size _start, .-_start

  .globl target
  .type target,@function
target:
  nop
.Ldata:
  .word 0xcd027057
  ret
  .size target, .-target

  .reloc 0, R_RISCV_NONE
