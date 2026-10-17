; REQUIRES: x86, plugins, examples
; UNSUPPORTED: target={{.*windows.*}}
; Plugins are currently broken on AIX, at least in the CI.
; XFAIL: target={{.*}}-aix{{.*}}

; RUN: opt %s -o %t.o
; RUN: ld.lld -shared --lto-emit-asm -%loadnewpmbye %t.o -o %t.inactive 2>&1 | count 0
; RUN: FileCheck %s --check-prefix=INACTIVE < %t.inactive.lto.s
; RUN: ld.lld -shared --lto-emit-asm -%loadnewpmbye -mllvm=%loadbye -mllvm=-last-words %t.o -o %t.active
; RUN: FileCheck %s --check-prefix=ACTIVE < %t.active.lto.s
; RUN: not ld.lld -shared -%loadnewpmbye -mllvm=%loadbye -mllvm=-last-words %t.o -o %t.err 2>&1 | FileCheck %s --check-prefix=ERR
; RUN: ld.lld -shared --lto-emit-asm --lto-partitions=2 -%loadnewpmbye -mllvm=%loadbye -mllvm=-last-words %t.o -o %t.parallel
; RUN: FileCheck %s --check-prefix=PARALLEL0 < %t.parallel.lto.s
; RUN: FileCheck %s --check-prefix=PARALLEL1 < %t.parallel.lto.1.s

; INACTIVE-NOT: Bye
; INACTIVE: f:
; ACTIVE: CodeGen Bye
; ERR: error: last words unsupported for binary output
; PARALLEL0: CodeGen Bye
; PARALLEL1: CodeGen Bye

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define i32 @f(i32 %x) {
  ret i32 %x
}
