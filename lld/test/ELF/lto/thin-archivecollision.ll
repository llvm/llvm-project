; RUN: opt -module-summary %s -o %t.o
; RUN: opt -module-summary %p/Inputs/thin1.ll -o %t.coll.o
; RUN: llvm-ar rcs %t1.a %t.coll.o
; RUN: opt -module-summary %p/Inputs/thin2.ll -o %t.coll.o
; RUN: llvm-ar rcsc %t2.a %t.coll.o

; RUN: ld.lld %t.o %t1.a %t2.a -o %t
; RUN: llvm-nm %t | FileCheck %s

; Check we handle this case correctly even in presence of --whole-archive.
; RUN: ld.lld %t.o --whole-archive %t1.a %t2.a -o %t
; RUN: llvm-nm %t | FileCheck %s

; CHECK: T _start
; CHECK: T blah
; CHECK: T foo

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-scei-ps4"

define i32 @_start() {
entry:
  %call = call i32 @foo(i32 23)
  %call1 = call i32 @blah(i32 37)
  ret i32 0
}

declare i32 @foo(i32) #1
declare i32 @blah(i32) #1
