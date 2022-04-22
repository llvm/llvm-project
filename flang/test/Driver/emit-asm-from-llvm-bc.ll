; Verify that the driver can consume LLVM BC files. The expected assembly is
; fairly generic (tested on AArch64 and X86_64), but we may need to tweak when
; testing on other platforms. Note that the actual output doesn't matter as
; long as it's in Assembly format.

;-------------
; RUN COMMANDS
;-------------
; RUN: rm -f %t.bc
; RUN: %flang_fc1 -emit-llvm-bc %s -o %t.bc
; RUN: %flang_fc1 -S -o - %t.bc | FileCheck %s
; RUN: rm -f %t.bc

; RUN: rm -f %t.bc
; RUN: %flang -c -emit-llvm %s -o %t.bc
; RUN: %flang -S -o - %t.bc | FileCheck %s
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
