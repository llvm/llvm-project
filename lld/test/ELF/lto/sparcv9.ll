; REQUIRES: sparc
;; Test we can infer the e_machine value EM_SPARCV9 from a bitcode file.

; RUN: llvm-as %s -o %t.bc
; RUN: ld.lld %t.bc -o %t
; RUN: llvm-readobj -h %t | FileCheck %s

; CHECK    Class: 64-bit
; CHECK:   DataEncoding: BigEndian
; CHECK: Machine: EM_SPARCV9

target datalayout = "E-m:e-i64:64-n32:64-S128"
target triple = "sparc64-unknown-elf"

define void @_start() {
  ret void
}
