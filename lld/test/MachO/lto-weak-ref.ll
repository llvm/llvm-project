; REQUIRES: x86

; RUN: rm -rf %t; split-file %s %t

; RUN: llvm-mc -filetype=obj -triple=x86_64-apple-darwin %t/dylib.s -o %t/dylib.o
; RUN: %lld -dylib -lSystem %t/dylib.o -o %t/dylib.dylib

;; As baseline, compile the .ll file to a real .o file and check behavior.
; RUN: llc -filetype=obj %t/weak-ref.ll -o %t/obj.o
; RUN: %lld -dylib -lSystem %t/obj.o %t/dylib.dylib -o %t/test.obj
; RUN: llvm-objdump --macho --syms %t/test.obj | FileCheck %s --check-prefixes=WEAK-REF

;; Check that we get the same behavior compiling the .ll file to a bitcode .o
;; file and linking that.
; RUN: opt -module-summary %t/weak-ref.ll -o %t/bitcode.o
; RUN: %lld -dylib -lSystem %t/bitcode.o %t/dylib.dylib -o %t/test.lto
; RUN: llvm-objdump --macho --syms %t/test.lto | FileCheck %s --check-prefixes=WEAK-REF

; WEAK-REF: SYMBOL TABLE:
; WEAK-REF: w      *UND* _my_weak_extern_function

;--- dylib.s

.globl	_my_weak_extern_function
_my_weak_extern_function:
  ret

;--- weak-ref.ll
target datalayout = "e-m:o-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.15.0"

declare extern_weak void @my_weak_extern_function()

; Function Attrs: noinline nounwind optnone ssp uwtable
define i32 @bar(i1 zeroext %0) {
entry:
  br i1 %0, label %if.then, label %if.end

if.then:                                          ; preds = %entry
  call void @my_weak_extern_function()
  br label %if.end

if.end:                                           ; preds = %if.then, %entry
  ret i32 0
}
