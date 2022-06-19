; RUN: rm -f %t.a
; RUN: llvm-as -o %t.o %s
; RUN: llvm-as -o %t2.o %S/Inputs/libcall-archive.ll
; RUN: llvm-ar rcs %t.a %t2.o
; RUN: wasm-ld -o %t %t.o %t.a
; RUN: obj2yaml %t | FileCheck %s

target datalayout = "e-m:e-p:32:32-p10:8:8-p20:8:8-i64:64-n32:64-S128"
target triple = "wasm32-unknown-unknown"

define void @_start(i8* %a, i8* %b) {
entry:
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* %a, i8* %b, i64 1024, i1 false)
  ret void
}

declare void @llvm.memcpy.p0i8.p0i8.i64(i8* nocapture, i8* nocapture, i64, i1)

; CHECK:       - Type:            CUSTOM
; CHECK-NEXT:    Name:            name
; CHECK-NEXT:    FunctionNames:
; CHECK-NEXT:      - Index:           0
; CHECK-NEXT:        Name:            _start
; CHECK-NEXT:      - Index:           1
; CHECK-NEXT:        Name:            memcpy
