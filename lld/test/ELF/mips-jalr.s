# REQUIRES: mips

## Check handling of the R_MIPS_JALR relocation.

# RUN: llvm-mc -filetype=obj -triple=mipsel-unknown-linux %s -o %t.o
# RUN: llvm-readelf -r %t.o | FileCheck -check-prefix=REL %s

# RUN: ld.lld %t.o -shared -o %t.so
# RUN: llvm-objdump -d --no-show-raw-insn %t.so | FileCheck --check-prefix=SO %s

# RUN: ld.lld %t.o --defsym=bar=__start -o %t.so
# RUN: llvm-objdump -d --no-show-raw-insn %t.so | FileCheck --check-prefix=EXE %s

# REL: R_MIPS_JALR   {{.*}} bar
# REL: R_MIPS_JALR   {{.*}} foo
# REL: R_MIPS_JALR   {{.*}} far

# SO: jalr  $25
# SO: bal   {{.*}} <foo>
# SO: jalr  $25

# SO: jr    $25
# SO: b     {{.*}} <foo>
# SO: jr    $25

# EXE: bal   {{.*}} <foo>
# EXE: bal   {{.*}} <foo>
# EXE: jalr  $25

# EXE: b     {{.*}} <foo>
# EXE: b     {{.*}} <foo>
# EXE: jr    $25

  .text
  .global bar
  .global __start
  .option pic2
far:
  .space 0x4fff0
__start:
foo:
  jal bar
  nop
  jal foo
  nop
  jal far
  nop
l1:
  jr $25
  .reloc l1, R_MIPS_JALR, bar
l2:
  jr $25
  .reloc l2, R_MIPS_JALR, foo
l3:
  jr $25
  .reloc l3, R_MIPS_JALR, far
