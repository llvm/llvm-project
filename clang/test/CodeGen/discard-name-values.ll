; REQUIRES: x86-registered-target
;; Textual IR keeps its names while parsing; -discard-value-names only drops
;; names created afterwards (here by the inliner).
; RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -O1 -emit-llvm %s -o - | FileCheck %s --check-prefixes=CHECK,KEEP
; RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -O1 -emit-llvm %s -discard-value-names -o - | FileCheck %s --check-prefixes=CHECK,DISCARD

; CHECK-LABEL: define {{.*}} @g(i32 {{.*}}%a)
; KEEP:          %y.i = add i32 %a, 1
; DISCARD:       %1 = add i32 %a, 1

target triple = "x86_64-unknown-linux-gnu"

define internal i32 @f(i32 %x) {
  %y = add i32 %x, 1
  ret i32 %y
}

define i32 @g(i32 %a) {
  %r = call i32 @f(i32 %a)
  ret i32 %r
}
