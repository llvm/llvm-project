; REQUIRES: xtensa
; Linking Xtensa bitcode selects EM_XTENSA from the target triple.
; RUN: llvm-as %s -o %t.bc
; RUN: ld.lld %t.bc -o %t
; RUN: llvm-readobj -h %t | FileCheck %s

; CHECK: Machine: EM_XTENSA

target triple = "xtensa-unknown-unknown"
target datalayout = "e-m:e-p:32:32-i8:8:32-i16:16:32-i64:64-n32"

define void @_start() {
  ret void
}
