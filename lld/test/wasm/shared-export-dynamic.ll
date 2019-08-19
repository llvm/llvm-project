; RUN: llc -relocation-model=pic -filetype=obj %s -o %t.o

; By default all `default` symbols should be exported
; RUN: wasm-ld -shared -o %t.wasm %t.o
; RUN: obj2yaml %t.wasm | FileCheck %s -check-prefix=DEFAULT
; DEFAULT: foo

; Verify that `--no-export-dynamic` works with `-shared`
; RUN: wasm-ld -shared --no-export-dynamic -o %t2.wasm %t.o
; RUN: obj2yaml %t2.wasm | FileCheck %s -check-prefix=NO-EXPORT
; NO-EXPORT-NOT: foo

target triple = "wasm32-unknown-emscripten"

define default i32 @foo() {
entry:
  ret i32 0
}
