; REQUIRES: x86
; RUN: rm -rf %t; split-file %s %t
; RUN: opt -module-summary %t/first.ll -o %t/first.o
; RUN: opt -module-summary %t/second.ll -o %t/second.o
; RUN: %lld -dylib -lSystem %t/first.o %t/second.o -o %t/12
; RUN: llvm-objdump --syms %t/12 | FileCheck %s --check-prefix=FIRST
; RUN: %lld -dylib -lSystem %t/second.o %t/first.o -o %t/21
; RUN: llvm-objdump --syms %t/21 | FileCheck %s --check-prefix=SECOND

; FIRST:  w    O __TEXT,first  _foo
; SECOND: w    O __TEXT,second _foo

#--- first.ll

target triple = "x86_64-apple-macosx10.15.0"
target datalayout = "e-m:o-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"

define linkonce void @foo() section "__TEXT,first" {
  ret void
}

#--- second.ll

target triple = "x86_64-apple-macosx10.15.0"
target datalayout = "e-m:o-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"

define linkonce void @foo() section "__TEXT,second" {
  ret void
}
