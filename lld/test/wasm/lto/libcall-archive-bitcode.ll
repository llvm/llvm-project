; REQUIRES: x86

; RUN: rm -rf %t && split-file %s %t && cd %t
; RUN: llvm-as main.ll -o main.o
; RUN: llvm-as puts.ll -o puts.o
; RUN: llvm-mc -filetype=obj -triple=wasm32-unknown-unknown printf.s -o printf.o
; RUN: llvm-ar rcs libc.a puts.o printf.o

;; Ensure that no printf->puts translation occurs during LTO because puts is in
;; bitcode, but was not brought into the link. This would fail the link by
;; extracting bitcode after LTO.
; RUN: wasm-ld -o out.wasm main.o libc.a
; RUN: obj2yaml out.wasm | FileCheck %s

;; Test the same behavior with lazy objects.
; RUN: wasm-ld -o out-lazy.wasm main.o --start-lib puts.o --end-lib printf.o
; RUN: obj2yaml out-lazy.wasm | FileCheck %s

;; Test that translation DOES occur when puts is extracted and brought into the link.
; RUN: wasm-ld -o out-extracted.wasm main.o puts.o printf.o
; RUN: obj2yaml out-extracted.wasm | FileCheck %s --check-prefix=EXTRACTED

; CHECK-NOT: Name: puts
; CHECK:     Name: printf

; EXTRACTED: Name: puts
; EXTRACTED-NOT: Name: printf

;--- puts.ll
target datalayout = "e-m:e-p:32:32-p10:8:8-p20:8:8-i64:64-n32:64-S128-ni:1:10:20"
target triple = "wasm32-unknown-unknown"

define i32 @puts(ptr nocapture readonly %0) noinline {
  call void asm sideeffect "", ""()
  ret i32 0
}

;--- printf.s
.globl printf
printf:
  .functype printf (i32, i32) -> (i32)
  i32.const 0
  end_function

;--- main.ll
target datalayout = "e-m:e-p:32:32-p10:8:8-p20:8:8-i64:64-n32:64-S128-ni:1:10:20"
target triple = "wasm32-unknown-unknown"

@str = constant [5 x i8] c"foo\0A\00"

define i32 @_start() {
  %call = call i32 (ptr, ...) @printf(ptr @str)
  ret i32 0
}

declare i32 @printf(ptr, ...)