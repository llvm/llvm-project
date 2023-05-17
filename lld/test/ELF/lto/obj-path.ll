; REQUIRES: x86
;; Test --lto-obj-path= for regular LTO.

; RUN: rm -rf %t && split-file %s %t && cd %t
; RUN: opt 1.ll -o 1.bc
; RUN: opt 2.ll -o 2.bc

; RUN: rm -f 4.o
; RUN: ld.lld --lto-obj-path=4.o -shared 1.bc 2.bc -o 3
; RUN: llvm-nm 3 | FileCheck %s --check-prefix=NM
; RUN: llvm-objdump -d 4.o | FileCheck %s
; RUN: ls 3* 4* | count 2

; RUN: rm -f 3 4.o
; RUN: ld.lld --thinlto-index-only=3.txt --lto-obj-path=4.o -shared 1.bc 2.bc -o 3
; RUN: llvm-objdump -d 4.o | FileCheck %s
; RUN: not ls 3

; NM: T f
; NM: T g

; CHECK: file format elf64-x86-64
; CHECK: <f>:
; CHECK: <g>:

;--- 1.ll
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

declare void @g(...)

define void @f() {
entry:
  call void (...) @g()
  ret void
}

;--- 2.ll
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define void @g() {
entry:
  ret void
}
