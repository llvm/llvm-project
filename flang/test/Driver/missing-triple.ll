; Verify that the module triple is overridden by the driver - even when the
; module triple is missing.
; NOTE: At the time of writing, the tested behaviour was consistent with Clang

;-------------
; RUN COMMANDS
;-------------
; RUN: %flang_fc1 -S %s -o - 2>&1 | FileCheck %s
; RUN: %flang -S  %s -o - 2>&1 | FileCheck %s

;----------------
; EXPECTED OUTPUT
;----------------
; CHECK: warning: overriding the module target triple with {{.*}}

;------
; INPUT
;------
define void @foo() {
  ret void
}
