; REQUIRES: systemz
;; Test we can infer the e_machine value EM_S390 from a bitcode file.

; RUN: llvm-as %s -o %t.o
; RUN: ld.lld %t.o -o %t
; RUN: llvm-readobj -h %t | FileCheck %s

; CHECK: Class: 64-bit
; CHECK: DataEncoding: BigEndian
; CHECK: Machine: EM_S390

target datalayout = "E-m:e-i1:8:16-i8:8:16-i64:64-f128:64-v128:64-a:8:16-n32:64"
target triple = "s390x-unknown-linux-gnu"

define void @_start() {
entry:
  ret void
}
