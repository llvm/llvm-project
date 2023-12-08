; REQUIRES: x86

; RUN: rm -rf %t && split-file %s %t
; RUN: llvm-as %t/a.ll -o %t/a.o
; RUN: llvm-profdata merge %t/cs.proftext -o %t/cs.profdata

;; Ensure lld generates warnings for profile cfg mismatch.
; RUN: not %lld -dylib --cs-profile-path=%t/cs.profdata %t/a.o -o /dev/null 2>&1 | FileCheck %s
; RUN: not %lld -dylib --cs-profile-path=%t/cs.profdata --pgo-warn-mismatch %t/a.o -o /dev/null 2>&1 | FileCheck %s

;; Ensure lld will not generate warnings for profile cfg mismatch.
; RUN: %lld -dylib --cs-profile-path=%t/cs.profdata --no-pgo-warn-mismatch %t/a.o -o /dev/null

; CHECK: function control flow change detected (hash mismatch) foo Hash = [[#]]

;--- a.ll
target triple = "x86_64-apple-darwin"
target datalayout = "e-m:o-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"

define i32 @foo() {
entry:
  ret i32 0
}

;--- cs.proftext
:csir
_foo
# Func Hash:
2277602155505015273
# Num Counters:
2
# Counter Values:
1
0
