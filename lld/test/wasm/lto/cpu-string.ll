; RUN: llvm-as %s -o %t.o

; RUN: wasm-ld %t.o -o %t.wasm
; RUN: obj2yaml %t.wasm | FileCheck %s

; CHECK: bulk-memory
; CHECK-NOT: multimemory

; RUN: wasm-ld -mllvm -mcpu=mvp %t.o -o %t.mvp.wasm
; RUN: obj2yaml %t.mvp.wasm | FileCheck --check-prefix=CHECK-MVP %s

; CHECK-MVP-NOT: bulk-memory
; CHECK-MVP-NOT: multimemory

; RUN: wasm-ld -mllvm -mcpu=bleeding-edge %t.o -o %t.mvp.wasm
; RUN: obj2yaml %t.mvp.wasm | FileCheck --check-prefix=CHECK-BLEEDING-EDGE %s

; CHECK-BLEEDING-EDGE: bulk-memory
; CHECK-BLEEDING-EDGE: multimemory

target datalayout = "e-m:e-p:32:32-p10:8:8-p20:8:8-i64:64-n32:64-S128-ni:1:10:20"
target triple = "wasm32-unknown-unknown"

define void @_start() #0 {
entry:
  ret void
}
