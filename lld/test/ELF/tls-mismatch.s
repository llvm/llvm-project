# REQUIRES: x86
# RUN: llvm-mc -filetype=obj -triple=x86_64 %s -o %t.o
# RUN: echo 'movq tls1@GOTTPOFF(%rip), %rax' | llvm-mc -filetype=obj -triple=x86_64 - -o %t1.o
# RUN: ld.lld %t1.o %t.o -o /dev/null
# RUN: ld.lld %t.o %t1.o -o /dev/null
# RUN: ld.lld --start-lib %t.o --end-lib %t1.o -o /dev/null
# RUN: ld.lld %t1.o --start-lib %t.o --end-lib -o /dev/null

## The TLS definition mismatches a non-TLS reference.
# RUN: echo '.type tls1,@object; movq tls1,%rax' | llvm-mc -filetype=obj -triple=x86_64 - -o %t2.o
# RUN: not ld.lld %t2.o %t.o -o /dev/null 2>&1 | FileCheck %s
## We fail to flag the swapped case.
# RUN: ld.lld %t.o %t2.o -o /dev/null

## We fail to flag the STT_NOTYPE reference. This usually happens with hand-written
## assembly because compiler-generated code properly sets symbol types.
# RUN: echo 'movq tls1,%rax' | llvm-mc -filetype=obj -triple=x86_64 - -o %t3.o
# RUN: ld.lld %t3.o %t.o -o /dev/null

## Overriding a TLS definition with a non-TLS definition does not make sense.
# RUN: not ld.lld --defsym tls1=42 %t.o -o /dev/null 2>&1 | FileCheck %s

## Part of PR36049: This should probably be allowed.
# RUN: not ld.lld --defsym tls1=tls2 %t.o -o /dev/null 2>&1 | FileCheck %s

## An undefined symbol in module-level inline assembly of a bitcode file
## is considered STT_NOTYPE. We should not error.
# RUN: echo 'target triple = "x86_64-pc-linux-gnu" \
# RUN:   target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128" \
# RUN:   module asm "movq tls1@GOTTPOFF(%rip), %rax"' | llvm-as - -o %t.bc
# RUN: ld.lld %t.o %t.bc -o /dev/null
# RUN: ld.lld %t.bc %t.o -o /dev/null

# CHECK: error: TLS attribute mismatch: tls1

.globl _start
_start:
  addl $1, %fs:tls1@TPOFF
  addl $2, %fs:tls2@TPOFF

.tbss
.globl tls1, tls2
  .space 8
tls1:
  .space 4
tls2:
  .space 4
