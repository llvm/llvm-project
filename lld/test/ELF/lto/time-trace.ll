; REQUIRES: x86
; RUN: llvm-as %s -o %t.o
; RUN: ld.lld -m elf_x86_64 -shared %t.o -o %t.so --plugin-opt=time-trace=%t.trace.json
; RUN: FileCheck --input-file=%t.trace.json %s
;; Print to stdout
; RUN: ld.lld -m elf_x86_64 -shared %t.o -o %t.so --plugin-opt=time-trace=- | \
; RUN: FileCheck %s

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; Make sure the content is correct
; CHECK: "traceEvents"
; Make sure LTO events are recorded
; CHECK-SAME: "name":"LTO"

define void @foo() {
  ret void
}
