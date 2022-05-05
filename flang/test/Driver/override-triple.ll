; Verify that the module triple is overridden by the driver - even in the presence
; of a module triple.
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
; For the triple to be overridden by the driver, it needs to be different to the host triple.
; Use a random string to guarantee that.
target triple = "invalid-triple"

define void @foo() {
  ret void
}
