# RUN: llvm-mc -triple riscv32 -mattr=+experimental-xqcibi,+xandesperf %s -filetype=obj -o %t.o
# RUN: not ld.lld -pie %t.o -o /dev/null 2>&1 | FileCheck %s

  .option exact

  qc.e.bgeui s0, 20, TARGET
# CHECK: error: {{.*}} unknown vendor-specific relocation (193) in vendor namespace "QUALCOMM" against symbol TARGET

  .global QUALCOMM
QUALCOMM:
  nop

  qc.e.bgeui s0, 20, TARGET
# CHECK: error: {{.*}} unknown vendor-specific relocation (193) in vendor namespace "QUALCOMM" against symbol TARGET

  nds.bbc t0, 7, TARGET
# CHECK: error: {{.*}} unknown vendor-specific relocation (241) in vendor namespace "ANDES" against symbol TARGET

  .global ANDES
ANDES:
  nop

  nds.bbs t0, 7, TARGET
# CHECK: error: {{.*}} unknown vendor-specific relocation (241) in vendor namespace "ANDES" against symbol TARGET

  .global TARGET
TARGET:
  nop
