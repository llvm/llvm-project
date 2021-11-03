;; We're intentionally testing fatal errors (for malformed input files), and
;; fatal errors aren't supported for testing when main is run twice.
; XFAIL: main-run-twice

; REQUIRES: x86
; RUN: llvm-as %s -o %t.o
; RUN: not %lld %t.o -o /dev/null 2>&1 | FileCheck %s

; CHECK: error: input module has no datalayout

; This bitcode file has no datalayout.
; Check that we error out producing a reasonable diagnostic.
target triple = "x86_64-apple-macosx10.15.0"

define void @_start() {
  ret void
}
