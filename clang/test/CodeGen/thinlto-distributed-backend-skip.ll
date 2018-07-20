; REQUIRES: x86-registered-target

; Check that ThinLTO backend respects "SkipModuleByDistributedBackend"
; flag which can be set by indexing.

; RUN: opt -thinlto-bc -o %t.o %s

; RUN: %clang_cc1 -triple x86_64-grtev4-linux-gnu \
; RUN:   -fthinlto-index=%S/Inputs/thinlto-distributed-backend-skip.bc \
; RUN:   -emit-llvm -o - -x ir %t.o | FileCheck %s

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-grtev4-linux-gnu"

; CHECK: "empty"
; CHECK: target triple =
; CHECK-NOT: @main
define i32 @main() {
entry:
  ret i32 0
}
