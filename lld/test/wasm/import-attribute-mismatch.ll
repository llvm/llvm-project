; RUN: llc -filetype=obj %s -o %t1.o
; RUN: llc -filetype=obj %S/Inputs/import-attributes.ll -o %t2.o
; RUN: not wasm-ld --export call_foo --allow-undefined -o %t.wasm %t1.o %t2.o 2>&1 | FileCheck %s

target triple = "wasm32-unknown-unknown-wasm"

define void @_start() {
  call void @foo();
  ret void
}

declare void @foo() #0

attributes #0 = { "wasm-import-module"="bar" }

; CHECK: wasm-ld: error: import module mismatch for symbol: foo
; CHECK: >>> defined as bar in {{.*}}1.o
; CHECK: >>> defined as baz in {{.*}}2.o
