; Test in default mode
; RUN: llc -filetype=obj %s -o %t.o
; RUN: not wasm-ld --export=missing -o %t.wasm %t.o 2>&1 | FileCheck -check-prefix=CHECK-ERROR %s
; RUN: wasm-ld --export=hidden_function -o %t.wasm %t.o
; RUN: obj2yaml %t.wasm | FileCheck %s

; Now test in Emscripten mode
; RUN: llc -filetype=obj %s -o %t.o -mtriple=wasm32-unknown-emscripten
; RUN: not wasm-ld --export=missing -o %t.wasm %t.o 2>&1 | FileCheck -check-prefix=CHECK-ERROR %s
; RUN: wasm-ld --export=hidden_function -o %t.wasm %t.o
; RUN: obj2yaml %t.wasm | FileCheck %s --check-prefixes=CHECK,EMSCRIPTEN

target triple = "wasm32-unknown-unknown"

@llvm.used = appending global [1 x i8*] [i8* bitcast (i32 ()* @used_function to i8*)], section "llvm.metadata"

; Not exported by default, but forced via commandline
define hidden i32 @hidden_function() local_unnamed_addr {
entry:
  ret i32 0
}

; Not exported by default
define i32 @default_function() local_unnamed_addr {
entry:
  ret i32 0
}

; Exported because its part of llvm.used
define i32 @used_function() local_unnamed_addr {
entry:
  ret i32 0
}

; Exported by default
define void @_start() local_unnamed_addr {
entry:
  ret void
}

; CHECK-ERROR: error: symbol exported via --export not found: missing

; CHECK-NOT: - Name: default_function

; CHECK:        - Type:            EXPORT
; CHECK-NEXT:     Exports:
; CHECK-NEXT:       - Name:            memory
; CHECK-NEXT:         Kind:            MEMORY
; CHECK-NEXT:         Index:           0
; CHECK-NEXT:       - Name:            hidden_function
; CHECK-NEXT:         Kind:            FUNCTION
; CHECK-NEXT:         Index:           0
; EMSCRIPTEN-NEXT:  - Name:            used_function
; EMSCRIPTEN-NEXT:    Kind:            FUNCTION
; EMSCRIPTEN-NEXT:    Index:           1
; CHECK-NEXT:       - Name:            _start
; CHECK-NEXT:         Kind:            FUNCTION
; CHECK-NEXT:         Index:           2
; CHECK-NEXT:   - Type:            CODE
