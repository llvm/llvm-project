; REQUIRES: x86-registered-target
; RUN: %clang_cc1 -triple x86_64-apple-darwin10 -S -o - %s | FileCheck %s

; RUN: %clang_cc1 -triple x86_64-pc-linux -S -o %t %s 2>&1 | \
; RUN: FileCheck --check-prefix=WARN %s
; WARN: warning: overriding the module target triple with x86_64-pc-linux
; RUN: FileCheck --check-prefix=LINUX %s < %t

target triple = "x86_64-apple-darwin10"

; CHECK: .globl _f0
; LINUX: .globl f0
define i32 @f0() nounwind ssp {
       ret i32 0
}
