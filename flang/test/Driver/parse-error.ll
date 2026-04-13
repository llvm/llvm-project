; This file is a valid LLVM IR file, but we force the driver to treat it as
; Fortran (with the `-x` flag). This way we verify that the driver
; correctly rejects invalid Fortran input.

; Input type is implicit (correctly assumed to be LLVM IR)
; RUN: %flang_fc1 -S %s -o -

; Input type is explicitly set as Fortran
; Verify that parsing errors are correctly reported by the driver
; Focuses on actions inheriting from the following:
; * PrescanAndSemaAction (-fsyntax-only)
; * PrescanAndParseAction (-fdebug-unparse-no-sema)
; RUN: not %flang_fc1 -fdebug-unparse-no-sema -x f95 %s 2>&1 | FileCheck %s --check-prefix=ERROR
; RUN: not %flang_fc1 -fsyntax-only %s -x f95 2>&1 | FileCheck %s --check-prefix=ERROR

; ERROR: Could not scan {{.*}}parse-error.ll

define void @foo() {
  ret void
}
