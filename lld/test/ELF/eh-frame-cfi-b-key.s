// REQUIRES: aarch64
// RUN: llvm-mc -filetype=obj -triple aarch64 %s -o %t.o
// RUN: ld.lld %t.o -o %t --icf=all --eh-frame-hdr

.globl _start
_start:
.cfi_startproc
.cfi_b_key_frame
.cfi_endproc
