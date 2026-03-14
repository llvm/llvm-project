; REQUIRES: x86
; RUN: llvm-as %S/Inputs/undef.ll -o %tundef.o
; RUN: llvm-as %s -o %tweakundef.o
; RUN: llvm-as %S/Inputs/archive-3.ll -o %tdef.o

;; Test that the lazy bitcode %tdef.o is fetched.
; RUN: ld.lld %tundef.o --start-lib %tdef.o --end-lib -shared -o %t.so
; RUN: llvm-nm %t.so | FileCheck %s

;; Test %tweakundef.o does not change STB_GLOBAL to STB_WEAK.
; RUN: ld.lld %tundef.o %tweakundef.o --start-lib %tdef.o --end-lib -shared -o %t.so
; RUN: llvm-nm %t.so | FileCheck %s

;; %tweakundef.o does not fetch %tdef.o but %tundef.o does.
; RUN: ld.lld --start-lib %tdef.o --end-lib %tweakundef.o %tundef.o -shared -o %t.so
; RUN: llvm-nm %t.so | FileCheck %s

; CHECK: T foo

target triple = "x86_64-unknown-linux-gnu"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

declare extern_weak void @foo()
