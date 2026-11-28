## Verifies that llvm-bolt handles a call to a non-zero address that lies
## outside of every section. processInterproceduralReferences() used to
## dereference the empty section lookup for such a reference.

# REQUIRES: system-linux

# RUN: llvm-mc -filetype=obj -triple x86_64-unknown-unknown %s -o %t.o
# RUN: ld.lld %t.o -o %t.exe
# RUN: llvm-bolt %t.exe -o %t.bolt 2>&1 | FileCheck %s
# RUN: llvm-objdump -d --no-show-raw-insn %t.bolt \
# RUN:   | FileCheck %s --check-prefix=CHECK-DIS

# CHECK-NOT: BOLT-ERROR
# CHECK: BOLT: 1 out of 1 functions were overwritten.
# CHECK-DIS: <_start>:
# CHECK-DIS: callq 0x1000

  .text
  .globl _start
  .type _start, %function
_start:
  .cfi_startproc
  callq 0x1000
  retq
  .cfi_endproc
  .size _start, .-_start
