; REQUIRES: x86
; RUN: rm -rf %t && mkdir %t && cd %t
; RUN: mkdir d e
; RUN: opt -module-summary %s -o b.bc
; RUN: mkdir -p %t1 %t2
; RUN: opt -module-summary %p/Inputs/thin1.ll -o d/coll.o
; RUN: opt -module-summary %p/Inputs/thin2.ll -o e/coll.o

; RUN: llvm-ar rcS d/a.a d/coll.o e/coll.o
; RUN: ld.lld b.bc d/a.a -o out --save-temps
; RUN: llvm-nm out | FileCheck %s
; RUN: llvm-nm out2.lto.o | FileCheck %s --check-prefix=MOD2
; RUN: llvm-nm out3.lto.o | FileCheck %s --check-prefix=MOD3

; Check we handle this case correctly even in presence of --whole-archive.
; RUN: rm out1.lto.o out2.lto.o out3.lto.o
; RUN: ld.lld b.bc --whole-archive d/a.a -o out --save-temps
; RUN: llvm-nm out | FileCheck %s
; RUN: ls out1.lto.o out2.lto.o out3.lto.o

; CHECK: T _start
; CHECK: T blah
; CHECK: T foo

; MOD2: T foo
; MOD3: T blah

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-scei-ps4"

define i32 @_start() {
entry:
  %call = call i32 @foo(i32 23)
  %call1 = call i32 @blah(i32 37)
  ret i32 0
}

declare i32 @foo(i32) #1
declare i32 @blah(i32) #1
