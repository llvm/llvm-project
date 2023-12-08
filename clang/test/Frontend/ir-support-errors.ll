; RUN: not %clang_cc1 -S -o - %s 2>&1 | FileCheck %s

target triple = "x86_64-apple-darwin10"

define i32 @f0() nounwind ssp {
; CHECK: {{.*}}ir-support-errors.ll:7:16: error: use of undefined value '%x'
       ret i32 %x
}
