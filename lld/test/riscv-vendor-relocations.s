# RUN: llvm-mc -triple riscv32 %s -filetype=obj -o %t.o
# RUN: not ld.lld -pie %t.o -o /dev/null 2>&1 | FileCheck %s

  .option exact

  .global TARGET
TARGET:
  nop

.global INVALID_VENDOR
.reloc 1f, R_RISCV_VENDOR, INVALID_VENDOR+0
.reloc 1f, R_RISCV_CUSTOM255, TARGET
1:
  nop

# CHECK: error: {{.*}} unknown vendor-specific relocation (255) in vendor namespace "INVALID_VENDOR" against symbol TARGET