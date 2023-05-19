# REQUIRES: x86
# RUN: rm -rf %t; split-file %s %t
# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-darwin19.0.0 %t/bad-function.s -o %t/bad-function.o
# RUN: not %lld -lSystem -dylib -lc++ %t/bad-function.o -o /dev/null 2>&1 | FileCheck %s
# CHECK: error: {{.*}}bad-function.o:(__compact_unwind+0x0) references section __data which is not in segment __TEXT
# CHECK: error: {{.*}}bad-function.o:(__compact_unwind+0x20) references section __data which is not in segment __TEXT

#--- bad-function.s
.data
_not_a_function:
  .cfi_startproc
  .cfi_personality 155, ___gxx_personality_v0
  .cfi_def_cfa_offset 16
  retq
  .cfi_endproc

_not_a_function_2:
  .cfi_startproc
  .cfi_personality 155, ___gxx_personality_v0
  .cfi_def_cfa_offset 16
  retq
  .cfi_endproc

.subsections_via_symbols
