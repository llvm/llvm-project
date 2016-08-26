# Check R_MIPS_GPREL32 relocation calculation if input file conatins
# non-zero GP0 value in the .reginfo section.

# RUN: llvm-mc -filetype=obj -triple=mips-unknown-linux %s -o %t.o
# RUN: ld.lld -r -o %t-r.o %t.o
# RUN: ld.lld -shared -o %t.so %t-r.o
# RUN: llvm-readobj -mips-reginfo %t-r.o %t.so | FileCheck %s
# RUN: llvm-objdump -s -t %t.so | FileCheck --check-prefix=DUMP %s

# REQUIRES: mips

# CHECK: {{.*}}mips-gprel32-relocs-gp0.s.tmp-r.o
# CHECK: GP: 0x7FF0
# CHECK: {{.*}}mips-gprel32-relocs-gp0.s.tmp.so
# CHECK: GP: 0x27FF0

# DUMP: Contents of section .rodata:
# DUMP:  0114 ffff0004 ffff0008
#             ^ 0x10004 + 0x7ff0 - 0x27ff0
#                      ^ 0x10008 + 0x7ff0 - 0x27ff0

# DUMP: SYMBOL TABLE:
# DUMP: 00010008         .text          00000000 bar
# DUMP: 00010004         .text          00000000 foo
# DUMP: 00027ff0         .got           00000000 .hidden _gp

  .text
  .global  __start
__start:
  lw      $t0,%call16(__start)($gp)
foo:
  nop
bar:
  nop

  .section .rodata, "a"
v:
  .gpword foo
  .gpword bar
