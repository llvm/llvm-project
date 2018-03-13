; RUN: llc -filetype=obj %s -o %t.o
; RUN: not wasm-ld --check-signatures --undefined _Z3fooi \
; RUN:     -o %t.wasm %t.o 2>&1 | FileCheck %s

; CHECK: error: undefined symbol: foo(int)

; RUN: not wasm-ld --check-signatures --no-demangle --undefined _Z3fooi \
; RUN:     -o %t.wasm %t.o 2>&1 |  FileCheck -check-prefix=CHECK-NODEMANGLE %s

; CHECK-NODEMANGLE: error: undefined symbol: _Z3fooi

target triple = "wasm32-unknown-unknown-wasm"

define hidden void @_start() local_unnamed_addr {
entry:
    ret void
}
