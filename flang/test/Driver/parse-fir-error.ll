; This file is a valid LLVM IR file, but we force the driver to treat it as
; FIR (with the `-x` flag). This way we verify that the driver
; correctly rejects invalid FIR input.

;----------
; RUN LINES
;----------
; Input type is implicit (correctly assumed to be LLVM IR)
; RUN: %flang_fc1 -S %s -o -

; Input type is explicitly set as FIR
; Verify that parsing errors are correctly reported by the driver
; RUN: not %flang_fc1 -S -x fir %s 2>&1 | FileCheck %s --check-prefix=ERROR
; RUN: not %flang_fc1 -S %s -x mlir 2>&1 | FileCheck %s --check-prefix=ERROR

; ERROR: error: unexpected character
; ERROR: error: Could not parse FIR

define void @foo() {
  ret void
}
