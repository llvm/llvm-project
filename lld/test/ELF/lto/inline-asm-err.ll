; REQUIRES: x86
; RUN: llvm-as %s -o %t.bc
; RUN: not ld.lld -shared %t.bc -o /dev/null 2>&1 | FileCheck %s --check-prefix=REGULAR

;; For regular LTO, the original module name is lost.
; REGULAR: error: ld-temp.o <inline asm>:1:2: invalid instruction mnemonic 'invalid'

; RUN: opt -module-summary %s -o %t.bc
; RUN: not ld.lld -shared %t.bc -o /dev/null 2>&1 | FileCheck %s --check-prefix=THIN

; THIN: error: {{.*}}.bc <inline asm>:1:2: invalid instruction mnemonic 'invalid'

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define void @foo() {
  call void asm "invalid", ""()
  ret void
}
