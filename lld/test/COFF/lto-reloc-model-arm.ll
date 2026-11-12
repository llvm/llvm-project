; REQUIRES: arm

;; ARMNT LTO uses the static relocation model. With PIC, the /GS cookie load
;; failed with "Cannot represent this expression".

; RUN: rm -rf %t && split-file %s %t
; RUN: opt -module-summary %t/gs.ll -o %t/gs.obj
; RUN: llvm-mc -triple=thumbv7-windows-msvc -filetype=obj %t/crt.s -o %t/crt.obj
; RUN: lld-link /dll /noentry /export:f /out:%t/gs.dll %t/gs.obj %t/crt.obj
; RUN: llvm-readobj --coff-basereloc %t/gs.dll | FileCheck %s

;; The absolute movw/movt load of __security_cookie needs a base relocation.
; CHECK:      BaseReloc [
; CHECK-NEXT:   Entry {
; CHECK-NEXT:     Type: ARM_MOV32(T)

;--- gs.ll
target triple = "thumbv7-pc-windows-msvc"

;; @use is in a native object, so LTO keeps the array and its stack protector.
declare void @use(ptr)

define dso_local arm_aapcs_vfpcc void @f() #0 {
entry:
  %buf = alloca [16 x i8], align 1
  call void @use(ptr %buf)
  ret void
}

attributes #0 = { sspstrong }

;--- crt.s
  .text
  .globl __security_check_cookie
  .p2align 1
__security_check_cookie:
  bx lr

  .globl use
  .p2align 1
use:
  bx lr

  .data
  .globl __security_cookie
  .p2align 2
__security_cookie:
  .long 0
