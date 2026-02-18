# REQUIRES: riscv
# RUN: llvm-mc -triple riscv32 %s -filetype=obj -o %t.o
# RUN: not ld.lld -pie %t.o -o /dev/null 2>&1 | FileCheck %s

  .option exact

  .global TARGET
TARGET:
  nop

INVALID_VENDOR:
.reloc ., R_RISCV_VENDOR, INVALID_VENDOR+0
.reloc ., R_RISCV_VENDOR, INVALID_VENDOR+0
.reloc ., R_RISCV_CUSTOM255, TARGET
  nop

# CHECK: error: {{.*}}:(.text+0x4): malformed consecutive R_RISCV_VENDOR relocations
# CHECK: error: {{.*}}:(.text+0x4): unknown vendor-specific relocation (255) in namespace 'INVALID_VENDOR' against symbol 'TARGET'

## The vendor symbol must be defined. If not, don't bother with a better diagnostic.
# CHECK: error: a.o:(.text1+0x0): unknown relocation (255) against symbol TARGET
# CHECK: error: undefined symbol: undef
.section .text1,"ax"
.reloc ., R_RISCV_VENDOR, undef
.reloc ., R_RISCV_CUSTOM255, TARGET
nop
