; RUN: llc -filetype=obj %s -o %t.o

target triple = "wasm32-unknown-unknown-wasm"

define hidden void @entry() local_unnamed_addr #0 {
entry:
  ret void
}

; RUN: not wasm-ld -o %t.exe 2>&1 | FileCheck -check-prefix=IN %s
; IN: error: no input files

; RUN: not wasm-ld %t.o 2>&1 | FileCheck -check-prefix=OUT %s
; OUT: error: no output file specified

; RUN: not wasm-ld 2>&1 | FileCheck -check-prefix=BOTH %s
; BOTH:     error: no input files
; BOTH-NOT: error: no output file specified
