;===----------------------------------------------------------------------===;
; This directory can be used to add Integration tests involving multiple
; stages of the compiler (for eg. from Fortran to LLVM IR). It should not
; contain executable tests. We should only add tests here sparingly and only
; if there is no other way to test. Repeat this message in each test that is
; added to this directory and sub-directories.
;===----------------------------------------------------------------------===;

; -fno-plt must not rewrite a loaded LLVM IR file. Clang leaves -x ir
; input unchanged, and Flang applies the flag only while lowering Fortran.
; -emit-llvm is required to check that the loaded module is not modified.

; RUN: %flang_fc1 -x ir -emit-llvm -fno-plt -o - %s | FileCheck %s

; The loaded functions are still emitted, and neither nonlazybind nor the
; RtLibUseGOT module flag is added anywhere in the output.
; CHECK-NOT: nonlazybind
; CHECK-NOT: RtLibUseGOT
; CHECK: define void @foo()
; CHECK-NOT: nonlazybind
; CHECK-NOT: RtLibUseGOT
; CHECK: declare void @bar()
; CHECK-NOT: nonlazybind
; CHECK-NOT: RtLibUseGOT

define void @foo() {
  call void @bar()
  ret void
}

declare void @bar()
