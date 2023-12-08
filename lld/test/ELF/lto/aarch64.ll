; REQUIRES: aarch64
;; Test we can infer the e_machine value EM_AARCH64 from a bitcode file.

; RUN: split-file %s %t
; RUN: llvm-as %t/le.s -o %t/le.o
; RUN: ld.lld %t/le.o -o %t/le
; RUN: llvm-readobj -h %t/le | FileCheck %s --check-prefixes=CHECK,LE

; RUN: llvm-as %t/be.s -o %t/be.o
; RUN: ld.lld %t/be.o -o %t/be
; RUN: llvm-readobj -h %t/be | FileCheck %s --check-prefixes=CHECK,BE

; CHECK:   Class: 64-bit
; LE:      DataEncoding: LittleEndian
; BE:      DataEncoding: BigEndian
; CHECK: Machine: EM_AARCH64

;--- le.s
target datalayout = "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128"
target triple = "aarch64-unknown-linux-gnu"

define void @_start() {
entry:
  ret void
}

;--- be.s
target datalayout = "E-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128"
target triple = "aarch64_be-unknown-linux-gnu"

define void @_start() {
entry:
  ret void
}
