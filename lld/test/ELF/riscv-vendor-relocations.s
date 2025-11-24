# REQUIRES: riscv
# RUN: llvm-mc -triple riscv32 %s -filetype=obj -o %t.o
# RUN: not ld.lld -pie %t.o -o /dev/null 2>&1 | FileCheck %s

  .option exact

  .global TARGET
TARGET:
  nop

.global INVALID_VENDOR
.global QUALCOMM
.global ANDES
1:
  nop

.reloc 1b, R_RISCV_VENDOR, INVALID_VENDOR+0
.reloc 1b, R_RISCV_VENDOR, INVALID_VENDOR+0
.reloc 1b, R_RISCV_CUSTOM255, TARGET
# CHECK: error: {{.*}}:(.text+0x4): malformed consecutive R_RISCV_VENDOR relocations
# CHECK: error: {{.*}}:(.text+0x4): unknown vendor-specific relocation (255) in namespace 'INVALID_VENDOR' against symbol 'TARGET'
.reloc 1b, R_RISCV_VENDOR, QUALCOMM+0
.reloc 1b, R_RISCV_CUSTOM192, TARGET
# CHECK: error: {{.*}}:(.text+0x4): unsupported vendor-specific relocation R_RISCV_QC_ABS20_U against symbol TARGET
.reloc 1b, R_RISCV_VENDOR, ANDES+0
.reloc 1b, R_RISCV_CUSTOM241, TARGET
# CHECK: error: {{.*}}:(.text+0x4): unsupported vendor-specific relocation R_RISCV_NDS_BRANCH_10 against symbol TARGET
2:
  nop
