# REQUIRES: x86
# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-darwin19.0.0 %s -o %t.o
# RUN: %lld -arch x86_64 -dylib %t.o -o %t.dylib
# RUN: llvm-objdump --macho --syms --unwind-info %t.dylib | FileCheck %s

## Both _f and _g have the same compact unwind encoding,
## but different stack sizes. So their compact unwindings
## can't be merged.
# CHECK:  SYMBOL TABLE:
# CHECK:  [[#%x,F:]] g  F __TEXT,__text _f
# CHECK:  [[#%x,G:]] g  F __TEXT,__text _g
# CHECK: Number of common encodings in array:       0x1
# CHECK:  Common encodings: (count = 1)
# CHECK:    encoding[0]: 0x03032000
# CHECK:  Second level indices:
# CHECK:    Second level index[0]:
# CHECK:      [0]: function offset=0x[[#%.8x,F]], encoding[0]=0x03032000
# CHECK:      [1]: function offset=0x[[#%.8x,G]], encoding[0]=0x03032000

## Based on compiling
##     int f() {
##       char alloca[3260] = { 0 };
##       return alloca[0];
##     }
##
##     int g() {
##       char alloca[2560] = { 0 };
##       return alloca[0];
##     }
## with `-fomit-frame-pointer -fno-stack-protector -S`.
.section __TEXT,__text,regular,pure_instructions
.build_version macos, 10, 15 sdk_version 10, 15, 6

.globl _f
.p2align 4, 0x90
_f:
  .cfi_startproc
  subq $3272, %rsp
  .cfi_def_cfa_offset 3280
  addq $3272, %rsp
  retq
  .cfi_endproc

.globl _g
.p2align 4, 0x90
_g:
  .cfi_startproc
  subq $2568, %rsp
  .cfi_def_cfa_offset 2576
  addq $2568, %rsp
  retq
  .cfi_endproc

.subsections_via_symbols
