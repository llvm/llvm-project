## Check that --print-only flag works with sections.

# REQUIRES: system-linux

# RUN: llvm-mc -filetype=obj -triple x86_64-unknown-linux %s -o %t.o
# RUN: ld.lld %t.o -o %t.exe
# RUN: llvm-bolt %t.exe -o %t.out --print-cfg --print-only=unused_code 2>&1 \
# RUN:   | FileCheck %s

# CHECK: Binary Function "foo"
# CHECK-NOT: Binary Function "_start"

  .text
  .globl _start
  .type _start, %function
_start:
  .cfi_startproc
  ret
  .cfi_endproc
  .size _start, .-_start

  .section unused_code,"ax",@progbits
  .globl foo
  .type foo, %function
foo:
  .cfi_startproc
  ret
  .cfi_endproc
  .size foo, .-foo
