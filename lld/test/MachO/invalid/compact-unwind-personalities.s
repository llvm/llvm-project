# REQUIRES: x86
# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-darwin19.0.0 %s -o %t.o
# RUN: not %lld -pie -lSystem -lc++ %t.o -o %t 2>&1 | FileCheck %s --check-prefix=TOO-MANY
# RUN: not %lld -pie -lSystem %t.o -o %t 2>&1 | FileCheck %s --check-prefix=UNDEF
# TOO-MANY: error: too many personalities (4) for compact unwind to encode
# UNDEF: error: undefined symbol: ___gxx_personality_v0

.globl _main, _personality_1, _personality_2, _personality_3

.text

_foo:
  .cfi_startproc
  .cfi_personality 155, _personality_1
  .cfi_def_cfa_offset 16
  retq
  .cfi_endproc

_bar:
  .cfi_startproc
  .cfi_personality 155, _personality_2
  .cfi_def_cfa_offset 16
  retq
  .cfi_endproc

_baz:
  .cfi_startproc
  .cfi_personality 155, _personality_3
  .cfi_def_cfa_offset 16
  retq
  .cfi_endproc

_main:
  .cfi_startproc
  .cfi_personality 155, ___gxx_personality_v0
  .cfi_def_cfa_offset 16
  retq
  .cfi_endproc

_personality_1:
  retq
_personality_2:
  retq
_personality_3:
  retq
