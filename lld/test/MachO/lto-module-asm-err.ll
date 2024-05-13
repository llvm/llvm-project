; REQUIRES: x86
; RUN: llvm-as %s -o %t.bc
; RUN: not %lld %t.bc -o /dev/null 2>&1 | FileCheck %s --check-prefix=REGULAR

;; For regular LTO, the original module name is lost.
; REGULAR: error: <inline asm>:2:1: invalid instruction mnemonic 'invalid'

; RUN: not opt -module-summary %s -o /dev/null 2>&1 | FileCheck %s --check-prefix=THIN

; THIN: error: <inline asm>:2:1: invalid instruction mnemonic 'invalid'

target triple = "x86_64-apple-macosx10.15.0"
target datalayout = "e-m:o-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"

module asm ".text"
module asm "invalid"

define void @main() {
  ret void
}
