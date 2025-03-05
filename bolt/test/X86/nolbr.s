## This reproduces a bug where profile collected from perf without LBRs and
## converted into fdata-no-lbr format is reported to not contain profile for any
## functions.

# REQUIRES: system-linux

# RUN: llvm-mc -filetype=obj -triple x86_64-unknown-unknown \
# RUN:   %s -o %t.o
# RUN: link_fdata --no-lbr %s %t.o %t.fdata
# RUN: FileCheck %s --input-file %t.fdata --check-prefix=CHECK-FDATA
# RUN: llvm-strip --strip-unneeded %t.o
# RUN: %clang %cflags %t.o -o %t.exe -Wl,-q -nostdlib
# RUN: llvm-bolt %t.exe -o %t.out --data %t.fdata --dyno-stats -nl \
# RUN:    --print-only=_start 2>&1 | FileCheck %s --check-prefix=CHECK-BOLT

# CHECK-FDATA:      no_lbr
# CHECK-FDATA-NEXT: 1 _start [[#]] 1

# CHECK-BOLT: BOLT-INFO: pre-processing profile using branch profile reader
# CHECK-BOLT: BOLT-INFO: operating with basic samples profiling data (no LBR).
# CHECK-BOLT: BOLT-INFO: 1 out of 1 functions in the binary (100.0%) have non-empty execution profile

  .globl _start
  .type _start, %function
_start:
  pushq	%rbp
  movq	%rsp, %rbp
  cmpl  $0x0, %eax
a:
# FDATA: 1 _start #a# 1
  je b
  movl	$0x0, %eax
  jmp c
b:
  movl  $0x1, %eax
c:
  popq	%rbp
  retq
.size _start, .-_start
