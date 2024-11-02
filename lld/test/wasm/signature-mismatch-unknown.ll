; RUN: llvm-mc -filetype=obj -triple=wasm32-unknown-unknown %p/Inputs/ret32.s -o %t.ret32.o
; RUN: llc -filetype=obj %s -o %t.main.o
; RUN: wasm-ld --fatal-warnings -o %t.wasm %t.ret32.o %t.main.o
; RUN: wasm-ld --fatal-warnings -o %t.wasm %t.main.o %t.ret32.o

; Also test the case where there are two different object files that contains
; references ret32:
; %t.main.o: Does not call ret32 directly; used the wrong signature.
; %t.call-ret32.o: Calls ret32 directly; uses the correct signature.
; RUN: llvm-mc -filetype=obj -triple=wasm32-unknown-unknown %p/Inputs/call-ret32.s -o %t.call-ret32.o
; RUN: wasm-ld --export=call_ret32 --fatal-warnings -o %t.wasm %t.main.o %t.call-ret32.o %t.ret32.o
; RUN: wasm-ld --export=call_ret32 --fatal-warnings -o %t.wasm %t.call-ret32.o %t.main.o %t.ret32.o

target triple = "wasm32-unknown-unknown"

; Function declaration with incorrect signature.
declare dso_local void @ret32()

; Simply taking the address of the function should *not* generate the
; the signature mismatch warning.
@ptr = dso_local global ptr @ret32, align 8

define hidden void @_start() local_unnamed_addr {
  %addr = load ptr, ptr @ptr, align 8
  call i32 %addr()
  ret void
}
