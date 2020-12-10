;; Test that weak undefined symbols do not fetch members from archive files.
; RUN: llc -filetype=obj %s -o %t.o
; RUN: llvm-mc -filetype=obj -triple=wasm32-unknown-unknown %p/Inputs/ret32.s -o %t.ret32.o
; RUN: llvm-mc -filetype=obj -triple=wasm32-unknown-unknown %p/Inputs/hello.s -o %t.hello.o
; RUN: rm -f %t.a
; RUN: llvm-ar rcs %t.a %t.ret32.o %t.hello.o

; RUN: wasm-ld %t.o %t.a -o %t.wasm
; RUN: obj2yaml %t.wasm | FileCheck %s

;; Also test with the library symbols being read first
; RUN: wasm-ld %t.a %t.o -o %t2.wasm
; RUN: obj2yaml %t2.wasm | FileCheck %s

; RUN: wasm-ld -u hello_str %t.o %t.a -o %t2.wasm
; RUN: obj2yaml %t2.wasm | FileCheck %s -check-prefix=CHECK-DATA

target triple = "wasm32-unknown-unknown"

; Weak external function symbol
declare extern_weak i32 @ret32()

; Weak external data symbol
@hello_str = extern_weak global i8*, align 4

define void @_start() {
  br i1 icmp ne (i8** @hello_str, i8** null), label %if.then, label %if.end

if.then:
  %call1 = call i32 @ret32()
  br label %if.end

if.end:
  ret void
}

; Ensure we have no data section.  If we do, would mean that hello_str was
; pulled out of the library.
; CHECK-NOT:  Type:            DATA
; CHECK-DATA: Type:            DATA

; CHECK: Name: 'undefined_weak:ret32'
; CHECK-NOT: Name: ret32
