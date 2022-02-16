; REQUIRES: x86
; RUN: rm -rf %t && split-file %s %t
; RUN: llvm-as %t/a.ll -o %t/a.bc
; RUN: llvm-mc -filetype=obj -triple=x86_64 %t/b.s -o %t/b.o
; RUN: llvm-ar rc %t/b.a %t/b.o
; RUN: ld.lld -shared --exclude-libs=b.a %t/a.bc %t/b.a -o %t.so -y __divti3 2>&1 | FileCheck %s --check-prefix=TRACE
; RUN: llvm-readelf --dyn-syms %t.so | FileCheck %s

; TRACE:      {{.*}}/b.a(b.o): lazy definition of __divti3
; TRACE-NEXT: lto.tmp: reference to __divti3
; TRACE-NEXT: {{.*}}/b.a(b.o): definition of __divti3

; CHECK:     Symbol table '.dynsym' contains 2 entries:
; CHECK-NOT: __divti3

;--- a.ll
target triple = "x86_64-unknown-linux"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"

define i128 @foo(i128 %x, i128 %y) {
entry:
  %div = sdiv i128 %x, %y
  ret i128 %div
}

;--- b.s
.globl __divti3
__divti3:
