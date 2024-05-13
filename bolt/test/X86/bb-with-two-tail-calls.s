# This reproduces a bug with dynostats when trying to compute branch stats
# at a block with two tails calls (one conditional and one unconditional).

# REQUIRES: system-linux

# RUN: llvm-mc -filetype=obj -triple x86_64-unknown-unknown \
# RUN:   %s -o %t.o
# RUN: link_fdata %s %t.o %t.fdata
# RUN: llvm-strip --strip-unneeded %t.o
# RUN: %clang %cflags %t.o -o %t.exe -Wl,-q -nostdlib
# RUN: llvm-bolt %t.exe -o %t.out --data %t.fdata --lite=0 --dyno-stats \
# RUN:    --print-sctc --print-only=_start 2>&1 | FileCheck %s
# CHECK-NOT: Assertion `BranchInfo.size() == 2 && "could only be called for blocks with 2 successors"' failed.
# Two tail calls in the same basic block after SCTC:
# CHECK:         {{.*}}:   ja      {{.*}} # TAILCALL  # CTCTakenCount: {{.*}}
# CHECK-NEXT:    {{.*}}:   jmp     {{.*}} # TAILCALL

  .globl _start
_start:
    je x
a:  ja b
    jmp c
x:  ret
# FDATA: 1 _start #a# 1 _start #b# 2 4
b:  jmp e
c:  jmp f

  .globl e
e:
    nop

  .globl f
f:
    nop
