; Verify that the driver can consume LLVM IR files. The expected assembly is
; fairly generic (verified on AArch64 and X86_64), but we may need to tweak when
; testing on other platforms. Note that the actual output doesn't matter
; as long as it's in Assembly format.

;-------------
; RUN COMMANDS
;-------------
; RUN: %flang_fc1 -S %s -o - | FileCheck %s
; RUN: %flang -S  %s -o - | FileCheck %s

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
