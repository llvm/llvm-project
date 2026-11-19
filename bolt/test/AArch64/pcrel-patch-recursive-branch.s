## Invalid external branch targets also cause recursive scans. A later ADR in
## the skipped function must still keep its target at the original address.

# REQUIRES: system-linux

# RUN: llvm-mc -triple=aarch64 -filetype=obj %s -o %t.o
# RUN: ld.lld --emit-relocs -e _start %t.o -o %t.exe
# RUN: llvm-bolt %t.exe -o %t.bolt --skip-funcs=_start --trap-old-code 2>&1 \
# RUN:   | FileCheck %s
# CHECK: BOLT-WARNING: corrupted control flow
# CHECK: BOLT-WARNING: unable to update PC-relative reference to second
# RUN: llvm-nm %t.exe | grep -w second > %t.sym
# RUN: llvm-nm %t.bolt | grep -w second | diff %t.sym -

  .text
  .type first, %function
first:
  ret
  .word 0
  .size first, .-first

  .type second, %function
second:
  ret
  .size second, .-second

  .globl _start
  .type _start, %function
_start:
  b first+4
  adr x0, second
  ret
  .size _start, .-_start

  .quad _start
