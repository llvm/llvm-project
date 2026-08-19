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

# CHECK: error: {{.*}}:(.text+0x4): R_RISCV_VENDOR is not followed by a relocation of code 192 to 255
# CHECK: error: {{.*}}:(.text+0x4): unknown vendor-specific relocation (255) in namespace 'INVALID_VENDOR' against symbol 'TARGET'

## R_RISCV_VENDOR followed by a standard relocation (not in 192-255 range).
# CHECK: error: {{.*}}:(.text1+0x0): R_RISCV_VENDOR is not followed by a relocation of code 192 to 255
.section .text1,"ax"
.reloc ., R_RISCV_VENDOR, INVALID_VENDOR
.reloc ., R_RISCV_32, TARGET
nop

## R_RISCV_VENDOR at end of section (no following relocation).
# CHECK: error: {{.*}}:(.text2+0x0): R_RISCV_VENDOR is not followed by a relocation of code 192 to 255
.section .text2,"ax"
.reloc ., R_RISCV_VENDOR, INVALID_VENDOR
nop

## Code 192 and 255 are in the valid range and reach the default case.
# CHECK: error: {{.*}}:(.text3+0x0): unknown vendor-specific relocation (192) in namespace 'INVALID_VENDOR' against symbol 'TARGET'
# CHECK: error: {{.*}}:(.text3+0x0): unknown vendor-specific relocation (255) in namespace 'INVALID_VENDOR' against symbol 'TARGET'
.section .text3,"ax"
.reloc ., R_RISCV_VENDOR, INVALID_VENDOR
.reloc ., R_RISCV_CUSTOM192, TARGET
.reloc ., R_RISCV_VENDOR, INVALID_VENDOR
.reloc ., R_RISCV_CUSTOM255, TARGET
nop

## The vendor symbol must be defined. If not, don't bother with a better diagnostic.
# CHECK: error: {{.*}}:(.text4+0x0): unknown relocation (255) against symbol TARGET
# CHECK: error: undefined symbol: undef
.section .text4,"ax"
.reloc ., R_RISCV_VENDOR, undef
.reloc ., R_RISCV_CUSTOM255, TARGET
nop
