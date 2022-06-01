; Verify that the driver can consume MLIR/FIR files.

;-------------
; RUN COMMANDS
;-------------
; RUN: %flang_fc1 -S %s -o - | FileCheck %s

;----------------
; EXPECTED OUTPUT
;----------------
; CHECK-LABEL: foo:
; CHECK: ret

;------
; INPUT
;------
func.func @foo() {
  return
}
