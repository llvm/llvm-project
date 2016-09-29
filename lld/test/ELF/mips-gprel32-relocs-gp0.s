# Check that relocatable object produced by LLD has zero gp0 value.
# Also check an error message if input object file has non-zero gp0 value.
# mips-gp0-non-zero.o is a relocatable object produced from the asm code
# below and linked by GNU bfd linker.

# RUN: llvm-mc -filetype=obj -triple=mips-unknown-linux %s -o %t.o
# RUN: ld.lld -r -o %t-r.o %t.o
# RUN: ld.lld -shared -o %t.so %t-r.o
# RUN: llvm-readobj -mips-reginfo %t-r.o %t.so | FileCheck %s
# RUN: llvm-objdump -s -t %t.so | FileCheck --check-prefix=DUMP %s
# RUN: not ld.lld -shared -o %t.so %S/Inputs/mips-gp0-non-zero.o 2>&1 \
# RUN:   | FileCheck --check-prefix=ERR %s

# REQUIRES: mips

# CHECK: {{.*}}mips-gprel32-relocs-gp0.s.tmp-r.o
# CHECK: GP: 0x0
# CHECK: {{.*}}mips-gprel32-relocs-gp0.s.tmp.so
# CHECK: GP: 0x27FF0

# DUMP: Contents of section .rodata:
# DUMP:  0114 fffe8014 fffe8018
#             ^ 0x10004 + 0 - 0x27ff0
#                      ^ 0x10008 + 0 - 0x27ff0

# DUMP: SYMBOL TABLE:
# DUMP: 00010008         .text          00000000 bar
# DUMP: 00010004         .text          00000000 foo
# DUMP: 00027ff0         .got           00000000 .hidden _gp

# ERR: {{.*}}mips-gp0-non-zero.o(.reginfo): unsupported non-zero ri_gp_value

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
