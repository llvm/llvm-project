; REQUIRES: riscv
;; Test we can infer the e_machine value EM_RISCV from a bitcode file.

; RUN: split-file %s %t
; RUN: llvm-as %t/32.ll -o %t/32.o
; RUN: ld.lld %t/32.o -o %t/32
; RUN: llvm-readobj -h %t/32 | FileCheck %s --check-prefixes=CHECK,RV32

; RUN: llvm-as %t/64.ll -o %t/64.o
; RUN: ld.lld %t/64.o -o %t/64
; RUN: llvm-readobj -h %t/64 | FileCheck %s --check-prefixes=CHECK,RV64

; RV32:    Class: 32-bit
; RV64:    Class: 64-bit
; CHECK:   DataEncoding: LittleEndian
; CHECK: Machine: EM_RISCV

;--- 32.ll
target datalayout = "e-m:e-p:32:32-i64:64-n32-S128"
target triple = "riscv32-unknown-elf"

define void @_start() {
  ret void
}

;--- 64.ll
target datalayout = "e-m:e-p:64:64-i64:64-i128:128-n64-S128"
target triple = "riscv64-unknown-elf"

define void @_start() {
  ret void
}
