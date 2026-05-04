; RUN: rm -rf %t && split-file %s %t && cd %t
; RUN: opt -thinlto-bc a.ll -o a.o
; RUN: opt -thinlto-bc b.ll -o b.o
; RUN: llvm-ar rcs b.a b.o
; RUN: opt -thinlto-bc c.ll -o c.o

;; Taking the address of the incorrectly declared @foo should not generate a warning.
; RUN: wasm-ld --fatal-warnings --no-entry --export-all a.o b.a -o a.out \
; RUN:         | FileCheck %s --implicit-check-not 'warning' --allow-empty

;; But we should still warn if we call the function with the wrong signature.
; RUN: not wasm-ld --fatal-warnings --no-entry --export-all a.o b.a c.o -o b.out 2>&1 \
; RUN:         | FileCheck %s --check-prefix=INVALID

; INVALID: error: function signature mismatch: foo
; INVALID: >>> defined as () -> void
; INVALID: >>> defined as () -> i32

;--- a.ll
target datalayout = "e-m:e-p:32:32-p10:8:8-p20:8:8-i64:64-n32:64-S128-ni:1:10:20"
target triple = "wasm32-unknown-unknown"

@ptr = constant ptr @foo
declare void @foo()

;--- b.ll
target datalayout = "e-m:e-p:32:32-p10:8:8-p20:8:8-i64:64-n32:64-S128-ni:1:10:20"
target triple = "wasm32-unknown-unknown"

define i32 @foo() noinline {
entry:
  ret i32 42
}

;--- c.ll
target datalayout = "e-m:e-p:32:32-p10:8:8-p20:8:8-i64:64-n32:64-S128-ni:1:10:20"
target triple = "wasm32-unknown-unknown"

declare void @foo()

define void @invalid() {
entry:
    call void @foo()
    ret void
}
