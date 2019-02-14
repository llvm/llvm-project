; RUN: llc -filetype=obj %p/Inputs/ret32.ll -o %t.ret32.o
; RUN: llc -filetype=obj -o %t.o %s
; RUN: wasm-ld -o %t.wasm %t.o %t.ret32.o -y ret32 -y _start 2>&1 | FileCheck %s -check-prefix=BOTH

; check alias
; RUN: wasm-ld -o %t.wasm %t.o %t.ret32.o -trace-symbol=_start 2>&1 | FileCheck %s -check-prefixes=JUST-START

target triple = "wasm32-unknown-unknown"

declare i32 @ret32(float %arg)

define void @_start() {
entry:
  %call1 = call i32 @ret32(float 0.0)
  ret void
}

; BOTH: .o: definition of _start
; BOTH: .o: reference to ret32
; BOTH: .ret32.o: definition of ret32

; JUST-START: .o: definition of _start
; JUST-START-NOT: ret32
