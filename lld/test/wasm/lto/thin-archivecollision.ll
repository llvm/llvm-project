; Based on lld/test/ELF/lto/thin-archivecollision.ll

; RUN: opt -module-summary %s -o %t.o
; RUN: mkdir -p %t1 %t2
; RUN: opt -module-summary %p/Inputs/thin1.ll -o %t1/t.coll.o
; RUN: opt -module-summary %p/Inputs/thin2.ll -o %t2/t.coll.o

; RUN: rm -f %t.a
; RUN: llvm-ar rcs %t.a %t1/t.coll.o %t2/t.coll.o
; RUN: wasm-ld %t.o %t.a -o %t
; RUN: obj2yaml %t | FileCheck %s

; Check we handle this case correctly even in presence of --whole-archive.
; RUN: wasm-ld %t.o --whole-archive %t.a -o %t
; RUN: obj2yaml %t | FileCheck %s

; CHECK: Name:            _start
; CHECK: Name:            foo
; CHECK: Name:            blah

target triple = "wasm32-unknown-unknown"

define i32 @_start() #0 {
entry:
  %call = call i32 @foo(i32 23)
  %call1 = call i32 @blah(i32 37)
  ret i32 0
}

declare i32 @foo(i32) #1
declare i32 @blah(i32) #1

attributes #0 = { noinline optnone }
