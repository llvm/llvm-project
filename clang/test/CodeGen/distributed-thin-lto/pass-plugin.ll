; REQUIRES: x86-registered-target, plugins, llvm-examples

;; Validate that -fpass-plugin works for distributed ThinLTO backends.

; RUN: opt -thinlto-bc -o %t.o %s

; RUN: llvm-lto2 run -thinlto-distributed-indexes %t.o \
; RUN:   -o %t2.index \
; RUN:   -r=%t.o,main,px

; RUN: %clang_cc1 -triple x86_64-grtev4-linux-gnu \
; RUN:   -O2 -emit-obj -fthinlto-index=%t.o.thinlto.bc \
; RUN:   -fpass-plugin=%llvmshlibdir/Bye%pluginext \
; RUN:   -mllvm -wave-goodbye \
; RUN:   -o %t.native.o -x ir %t.o 2>&1 | FileCheck %s

; CHECK: Bye: main

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-grtev4-linux-gnu"

define i32 @main() {
entry:
  ret i32 0
}
