; REQUIRES: x86-registered-target

; Trivial test for distributes ThinLTO

; RUN: opt -thinlto-bc -o %t.o %s

; RUN: llvm-lto2 run -thinlto-distributed-indexes %t.o \
; RUN:   -o %t2.index \
; RUN:   -r=%t.o,main,px

; RUN: %clang_cc1 -triple x86_64-grtev4-linux-gnu \
; RUN:   -emit-obj -fthinlto-index=%t.o.thinlto.bc \
; RUN:   -o %t.native.o -x ir %t.o

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-grtev4-linux-gnu"

define i32 @main() {
entry:
  ret i32 0
}
