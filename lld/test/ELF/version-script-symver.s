# REQUIRES: x86

# RUN: llvm-mc -filetype=obj -triple=x86_64-unknown-linux %s -o %t.o
# RUN: echo "VERSION { global: *; };" > %t.map
# RUN: ld.lld %t.o --version-script %t.map -o %t

.global _start
.global bar
.symver _start, bar@@VERSION
_start:
  jmp bar
