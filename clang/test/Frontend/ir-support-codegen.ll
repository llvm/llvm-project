; REQUIRES: x86-64-registered-target
; RUN: %clang_cc1 -S -o - %s | FileCheck %s

target triple = "x86_64-apple-darwin10"

; CHECK: .globl _f0
define i32 @f0() nounwind ssp {
       ret i32 0
}
