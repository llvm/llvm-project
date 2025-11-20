; REQUIRES: arm
;; Test we can infer the e_machine value EM_ARM from a bitcode file.

; RUN: split-file %s %t

; RUN: llvm-as %t/arm.ll -o %t/arm.o
; RUN: ld.lld %t/arm.o -o %t/arm
; RUN: llvm-readobj -h %t/arm | FileCheck %s --check-prefix=ARM
; RUN: llvm-as %t/armeb.ll -o %t/armeb.o
; RUN: ld.lld %t/armeb.o -o %t/armeb
; RUN: llvm-readobj -h %t/armeb | FileCheck %s --check-prefix=ARMEB

; RUN: llvm-as %t/thumb.ll -o %t/thumb.o
; RUN: ld.lld %t/thumb.o -o %t/thumb
; RUN: llvm-readobj -h %t/thumb | FileCheck %s --check-prefix=THUMB
; RUN: llvm-as %t/thumbeb.ll -o %t/thumbeb.o
; RUN: ld.lld %t/thumbeb.o -o %t/thumbeb
; RUN: llvm-readobj -h %t/thumbeb | FileCheck %s --check-prefix=THUMBEB

; ARM:       Class: 32-bit
; ARM:       DataEncoding: LittleEndian
; ARM:     Machine: EM_ARM (
; ARMEB:     Class: 32-bit
; ARMEB:     DataEncoding: BigEndian
; ARMEB:   Machine: EM_ARM (

; THUMB:     Class: 32-bit
; THUMB:     DataEncoding: LittleEndian
; THUMB:   Machine: EM_ARM (
; THUMBEB:   Class: 32-bit
; THUMBEB:   DataEncoding: BigEndian
; THUMBEB: Machine: EM_ARM (

;--- arm.ll
target datalayout = "e-m:e-p:32:32-Fi8-i64:64-v128:64:128-a:0:32-n32-S64"
target triple = "armv7-linux-musleabi"

define void @_start() {
  ret void
}

;--- thumb.ll
target datalayout = "e-m:e-p:32:32-Fi8-i64:64-v128:64:128-a:0:32-n32-S64"
target triple = "thumbv8m.base-none-unknown-gnueabi"

define void @_start() {
  ret void
}

;--- armeb.ll
target datalayout = "E-m:e-p:32:32-Fi8-i64:64-v128:64:128-a:0:32-n32-S64"
target triple = "armebv7-unknown-linux-musleabi"

define void @_start() {
  ret void
}

;--- thumbeb.ll
target datalayout = "E-m:e-p:32:32-Fi8-i64:64-v128:64:128-a:0:32-n32-S64"
target triple = "thumbebv8m.base-none-unknown-gnueabi"

define void @_start() {
  ret void
}
