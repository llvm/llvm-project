## Scanning first recursively scans third, which references first again.
## Each scan must retain its own symbolizer for the references that follow.

# REQUIRES: system-linux

# RUN: llvm-mc -triple=aarch64 -filetype=obj %s -o %t.o
# RUN: ld.lld --emit-relocs -e _start %t.o -o %t.exe
# RUN: llvm-bolt %t.exe -o %t.bolt --skip-funcs=_start --trap-old-code
# RUN: llvm-nm %t.exe | grep -E ' t (first|second|third|fourth)$' > %t.sym
# RUN: llvm-nm %t.bolt | grep -E ' t (first|second|third|fourth)$' | diff %t.sym -

  .text
  .type third, %function
third:
  adr x4, first
  ret
  .size third, .-third

  .type fourth, %function
fourth:
  ret
  .size fourth, .-fourth

  .type first, %function
first:
  adr x2, third
  adr x3, fourth
  ret
  .size first, .-first

  .type second, %function
second:
  mov w0, #42
  mov w8, #93
  svc #0
  .size second, .-second

  .globl _start
  .type _start, %function
_start:
  adr x1, first
  adr x0, second
  br x0
  .size _start, .-_start

  .quad _start
