# REQUIRES: x86
# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-darwin19.0.0 %s -o %t.o
# RUN: not %lld -pie -lSystem -lc++ %t.o -o %t 2>&1 | FileCheck %s -DFILE=%t.o
# CHECK: error: compact unwind references address in [[FILE]]:(__data) which is not in segment __TEXT

.globl _main, _not_a_function
.text
_main:
  retq

.data
_not_a_function:
  .cfi_startproc
  .cfi_personality 155, ___gxx_personality_v0
  .cfi_def_cfa_offset 16
  retq
  .cfi_endproc
