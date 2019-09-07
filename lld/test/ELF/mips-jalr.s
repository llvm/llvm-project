# REQUIRES: mips
# Check that lld ignores R_MIPS_JALR relocation for now.

# RUN: llvm-mc -filetype=obj -triple=mips-unknown-linux %s -o %t.o
# RUN: ld.lld %t.o -o %t.exe 
# RUN: llvm-readelf -r %t.o | FileCheck -check-prefix=REL %s
# RUN: llvm-objdump -d --no-show-raw-insn %t.exe | FileCheck %s

# REL: R_MIPS_CALL16 {{.*}} foo
# REL: R_MIPS_JALR   {{.*}} foo

# CHECK: jalr  $25

  .text
  .global  __start
  .option pic2
__start:
  jal foo
foo:
  nop
