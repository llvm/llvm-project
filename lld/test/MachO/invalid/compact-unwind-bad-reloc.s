# REQUIRES: x86
# RUN: rm -rf %t; split-file %s %t
# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-darwin19.0.0 %t/bad-function.s -o %t/bad-function.o
# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-darwin19.0.0 %t/bad-personality.s -o %t/bad-personality.o
# RUN: not %lld -lSystem -lc++ %t/bad-function.o -o %t 2>&1 | FileCheck %s -DFILE=%t/bad-function.o -D#OFF=0x0
# RUN: not %lld -lSystem -lc++ %t/bad-personality.o -o %t 2>&1 | FileCheck %s -DFILE=%t/bad-personality.o -D#OFF=0x10
# CHECK: error: [[FILE]]:(__compact_unwind+0x[[#%x,OFF]]) references section __data which is not in segment __TEXT

#--- bad-function.s
.data
_not_a_function:
  .cfi_startproc
  .cfi_personality 155, ___gxx_personality_v0
  .cfi_def_cfa_offset 16
  retq
  .cfi_endproc

#--- bad-personality.s
.globl _main, _not_a_function
.text
_main:
  .cfi_startproc
  .cfi_personality 155, _my_personality
  .cfi_def_cfa_offset 16
  retq
  .cfi_endproc

.data
.globl _my_personality
_my_personality:
