; RUN: llc -filetype=obj %p/Inputs/weak-symbol1.ll -o %t.weak.o
; RUN: llc -filetype=obj %p/Inputs/strong-symbol.ll -o %t.strong.o
; RUN: llc -filetype=obj %s -o %t.o
; RUN: not wasm-ld --check-signatures -o %t.wasm %t.o %t.strong.o %t.weak.o 2>&1 | FileCheck %s
; RUN: wasm-ld -o %t.wasm %t.o %t.strong.o %t.weak.o

target triple = "wasm32-unknown-unknown-wasm"

declare i32 @weakFn() local_unnamed_addr

define void @_start() local_unnamed_addr {
entry:
  %call = call i32 @weakFn()
  ret void
}

; CHECK: error: Function type mismatch: weakFn
; CHECK-NEXT: >>> defined as () -> I32 in {{.*}}signature-mismatch-weak.ll.tmp.o
; CHECK-NEXT: >>> defined as () -> I64 in {{.*}}signature-mismatch-weak.ll.tmp.strong.o
