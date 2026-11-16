# REQUIRES: x86
# RUN: rm -rf %t && split-file %s %t && cd %t
# RUN: llvm-mc -filetype=obj -triple=x86_64 a.s -o %t.o
# RUN: echo 'movq tls1@GOTTPOFF(%rip), %rax' | llvm-mc -filetype=obj -triple=x86_64 - -o %t1.o
# RUN: ld.lld %t1.o %t.o -o /dev/null
# RUN: ld.lld %t.o %t1.o -o /dev/null
# RUN: ld.lld --start-lib %t.o --end-lib %t1.o -o /dev/null
# RUN: ld.lld %t1.o --start-lib %t.o --end-lib -o /dev/null

## The TLS definition mismatches a non-TLS reference.
# RUN: echo '.type tls1,@object; movq tls1,%rax' | llvm-mc -filetype=obj -triple=x86_64 - -o %t2.o
# RUN: not ld.lld %t2.o %t.o -o /dev/null 2>&1 | FileCheck %s
# RUN: not ld.lld %t.o %t2.o -o /dev/null 2>&1 | FileCheck %s

## Non-TLS definition mismatches a TLS reference.
# RUN: llvm-mc -filetype=obj -triple=x86_64 bad-type.s -o bad-type.o
# RUN: not ld.lld bad-type.o %t1.o 2>&1 | FileCheck %s --check-prefix=CHECK2
# RUN: not ld.lld %t1.o bad-type.o 2>&1 | FileCheck %s --check-prefix=CHECK2

## We fail to flag the STT_NOTYPE reference. This usually happens with hand-written
## assembly because compiler-generated code properly sets symbol types.
# RUN: echo 'movq tls1,%rax' | llvm-mc -filetype=obj -triple=x86_64 - -o %t3.o
# RUN: ld.lld %t3.o %t.o -o /dev/null

## Overriding a TLS definition with a non-TLS definition does not make sense.
## We fail to flag this case.
# RUN: ld.lld --defsym tls1=42 %t.o -o /dev/null 2>&1 | count 0

## Part of PR36049: This should probably be allowed.
# RUN: ld.lld --defsym tls1=tls2 %t.o -o /dev/null 2>&1 | count 0

## An undefined symbol in module-level inline assembly of a bitcode file
## is considered STT_NOTYPE. We should not error.
# RUN: echo 'target triple = "x86_64-pc-linux-gnu" \
# RUN:   target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128" \
# RUN:   module asm "movq tls1@GOTTPOFF(%rip), %rax"' | llvm-as - -o %t.bc
# RUN: ld.lld %t.o %t.bc -o /dev/null
# RUN: ld.lld %t.bc %t.o -o /dev/null

## Don't error when a bitcode file defines the TLS symbol in module-level assembly (STT_NOTYPE before LTO).
# RUN: llvm-as tls-def.ll -o tls-def.bc
# RUN: ld.lld %t1.o tls-def.bc
# RUN: ld.lld tls-def.bc %t1.o

## -u creates an Undefined with STT_NOTYPE. The object file's STT_TLS undefined
## should update the type so that the TLS IE GOT entry is correctly created.
# RUN: ld.lld -shared -u tls1 %t1.o %t1.o -o %t1.so
# RUN: llvm-readelf -rs %t1.so | FileCheck %s --check-prefix=UNDEF
# UNDEF: R_X86_64_TPOFF64 {{.*}} tls1 + 0
# UNDEF: 0000000000000000     0 TLS     GLOBAL DEFAULT  UND tls1

# CHECK: error: TLS attribute mismatch: tls1
# CHECK-NEXT: >>> in {{.*}}.tmp.o
# CHECK-NEXT: >>> in {{.*}}

# CHECK2: error: TLS attribute mismatch: tls1
# CHECK2-NEXT: >>> in bad-type.o
# CHECK2-NEXT: >>> in {{.*}}.tmp1.o

#--- a.s
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

#--- bad-type.s
.globl _start, tls1
_start:
tls1:

#--- tls-def.ll
target triple = "x86_64-unknown-linux-gnu"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"

module asm ".globl _start, tls1"
module asm "_start:"
module asm ".section .tbss,\22awT\22,@nobits"
module asm "tls1: .long 0"
