; RUN: llc -filetype=obj %s -o %t.o
; RUN: lld -flavor wasm --allow-undefined -o %t.wasm %t.o

; Fails due to undefined 'foo'
; RUN: not lld -flavor wasm -o %t.wasm %t.o 2>&1 | FileCheck %s
; CHECK: error: {{.*}}.o: undefined symbol: foo

; But succeeds if we pass a file containing 'foo' as --allow-undefined-file.
; RUN: echo 'foo' > %t.txt
; RUN: lld -flavor wasm --allow-undefined-file=%t.txt -o %t.wasm %t.o

target triple = "wasm32-unknown-unknown-wasm"

; Takes the address of the external foo() resulting in undefined external
@bar = hidden local_unnamed_addr global i8* bitcast (i32 ()* @foo to i8*), align 4

declare i32 @foo() #0

define hidden void @_start() local_unnamed_addr #0 {
entry:
    ret void
}
