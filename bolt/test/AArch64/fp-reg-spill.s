# Check that we correctly handle arm64 fp register spills in
# bolt when we are processing jump tables.
# REQUIRES: system-linux
# RUN: llvm-mc -filetype=obj -triple=aarch64 %s -o %t.o
# RUN: ld.lld --emit-relocs %t.o -o %t.elf
# RUN: llvm-bolt --jump-tables=move %t.elf -o %t.bolt

.globl _foo, _start

_foo:
  ret

_start:
  adr x6, _foo
  fmov d18,x6
  fmov x5,d18
  ldrb  w5, [x5, w1, uxtw]
  add x5, x6, w5, sxtb #2
  br x5
