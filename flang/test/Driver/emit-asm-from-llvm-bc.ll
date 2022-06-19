; Verify that the driver can consume LLVM BC files.

; REQUIRES: aarch64-registered-target

;-------------
; RUN COMMANDS
;-------------
; RUN: rm -f %t.bc
; RUN: %flang_fc1 -triple aarch64-unknown-linux-gnu -emit-llvm-bc %s -o %t.bc
; RUN: %flang_fc1 -S -triple aarch64-unknown-linux-gnu -o - %t.bc | FileCheck %s
; RUN: rm -f %t.bc

; RUN: rm -f %t.bc
; RUN: %flang -c -target aarch64-unknown-linux-gnu -emit-llvm %s -o %t.bc
; RUN: %flang -S -target aarch64-unknown-linux-gnu -o - %t.bc | FileCheck %s
; RUN: rm -f %t.bc

;----------------
; EXPECTED OUTPUT
;----------------
; CHECK-LABEL: foo:
; CHECK: ret

;------
; INPUT
;------
define void @foo() {
  ret void
}
