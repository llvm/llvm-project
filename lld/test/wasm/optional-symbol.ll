; RUN: llc -filetype=obj -o %t.o %s
; RUN: wasm-ld --export=get_handle %t.o -o %t.wasm

target triple = "wasm32-unknown-unknown"

@__dso_handle = external global i8*

define i8** @get_handle() {
  ret i8** @__dso_handle
}

define void @_start() {
  ret void
}
