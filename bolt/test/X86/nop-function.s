## Check that BOLT preserves nop instruction if it's the only instruction
## in a function.

# REQUIRES: system-linux

# RUN: llvm-mc -filetype=obj -triple x86_64-unknown-linux %s -o %t.o
# RUN: ld.lld %t.o -o %t.exe -q
# RUN: llvm-bolt %t.exe -o %t.bolt.exe --relocs=0
# RUN: llvm-objdump -d %t.bolt.exe | FileCheck %s

  .text
  .globl nop_function
  .type nop_function,@function
nop_function:
  .cfi_startproc
  nop
# CHECK: <nop_function>:
# CHECK-NEXT: nop

  .size nop_function, .-nop_function
  .cfi_endproc


  .globl _start
  .type _start,@function
_start:
  .cfi_startproc
  call nop_function
  .size _start, .-_start
  .cfi_endproc
