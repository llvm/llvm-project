; REQUIRES: ppc
;; Test we can infer the e_machine value EM_PPC/EM_PPC64 from a bitcode file.

; RUN: split-file %s %t
; RUN: llvm-as %t/32le.ll -o %t/32le.o
; RUN: ld.lld %t/32le.o -o %t/32le
; RUN: llvm-readobj -h %t/32le | FileCheck %s --check-prefix=LE32

; RUN: llvm-as %t/32be.ll -o %t/32be.o
; RUN: ld.lld %t/32be.o -o %t/32be
; RUN: llvm-readobj -h %t/32be | FileCheck %s --check-prefix=BE32

; RUN: llvm-as %t/64.ll -o %t/64.o
; RUN: ld.lld %t/64.o -o %t/64
; RUN: llvm-readobj -h %t/64 | FileCheck %s --check-prefix=LE64

; LE32:   Class: 32-bit
; LE32:   DataEncoding: LittleEndian
; LE32: Machine: EM_PPC (

; BE32:   Class: 32-bit
; BE32:   DataEncoding: BigEndian
; BE32: Machine: EM_PPC (

; LE64:   Class: 64-bit
; LE64:   DataEncoding: LittleEndian
; LE64: Machine: EM_PPC64

;--- 32le.ll
target datalayout = "e-m:e-p:32:32-Fn32-i64:64-n32"
target triple = "powerpcle-pc-freebsd"

define void @_start() {
  ret void
}

;--- 32be.ll
target datalayout = "E-m:e-p:32:32-Fn32-i64:64-n32"
target triple = "powerpc-unknown-linux-gnu"

define void @_start() {
  ret void
}

;--- 64.ll
target datalayout = "e-m:e-Fn32-i64:64-n32:64-S128-v256:256:256-v512:512:512"
target triple = "powerpc64le-unknown-linux-gnu"

define void @_start() {
  ret void
}
