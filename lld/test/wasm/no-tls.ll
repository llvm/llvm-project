; Testing that __tls_size and __tls_align are correctly emitted when there are
; no thread_local variables.

; RUN: llc -mattr=+bulk-memory,+atomics -filetype=obj %s -o %t.o

target triple = "wasm32-unknown-unknown"

define void @_start() local_unnamed_addr {
entry:
  ret void
}

; RUN: wasm-ld -no-gc-sections --shared-memory --max-memory=131072 --allow-undefined -o %t.wasm %t.o
; RUN: obj2yaml %t.wasm | FileCheck %s
; CHECK:       - Type:            GLOBAL
; CHECK-NEXT:    Globals:

; __stack_pointer
; CHECK-NEXT:      - Index:           0
; CHECK-NEXT:        Type:            I32
; CHECK-NEXT:        Mutable:         true
; CHECK-NEXT:        InitExpr:
; CHECK-NEXT:          Opcode:          I32_CONST
; CHECK-NEXT:          Value:           66576

; __tls_base
; CHECK-NEXT:      - Index:           1
; CHECK-NEXT:        Type:            I32
; CHECK-NEXT:        Mutable:         true
; CHECK-NEXT:        InitExpr:
; CHECK-NEXT:          Opcode:          I32_CONST
; CHECK-NEXT:          Value:           0

; __tls_size
; CHECK-NEXT:      - Index:           2
; CHECK-NEXT:        Type:            I32
; CHECK-NEXT:        Mutable:         false
; CHECK-NEXT:        InitExpr:
; CHECK-NEXT:          Opcode:          I32_CONST
; CHECK-NEXT:          Value:           0

; __tls_align
; CHECK-NEXT:      - Index:           3
; CHECK-NEXT:        Type:            I32
; CHECK-NEXT:        Mutable:         false
; CHECK-NEXT:        InitExpr:
; CHECK-NEXT:          Opcode:          I32_CONST
; CHECK-NEXT:          Value:           1
