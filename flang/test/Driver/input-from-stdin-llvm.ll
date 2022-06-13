; Verify that reading from stdin works as expected - LLVM input

; REQUIRES: aarch64-registered-target

;----------
; RUN LINES
;----------
; Input type is implicit - assumed to be Fortran. As the input is provided via
; stdin, the file extension is not relevant here.
; RUN: cat %s | not %flang -S - -o -
; RUN: cat %s | not %flang_fc1 -S - -o -

; Input type is explicit
; RUN: cat %s | %flang -x ir -S -target aarch64-unknown-linux-gnu - -o - | FileCheck %s
; RUN: cat %s | %flang_fc1 -x ir -S -triple aarch64-unknown-linux-gnu - -o - | FileCheck %s

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
