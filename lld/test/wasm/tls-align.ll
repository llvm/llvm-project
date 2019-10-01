; RUN: llc -mattr=+bulk-memory,+atomics -filetype=obj %s -o %t.o

target triple = "wasm32-unknown-unknown"

@no_tls = global i32 0, align 4
@tls1 = thread_local(localexec) global i32 1, align 4
@tls2 = thread_local(localexec) global i32 1, align 16

define i32* @tls1_addr() {
  ret i32* @tls1
}

define i32* @tls2_addr() {
  ret i32* @tls2
}

; RUN: wasm-ld -no-gc-sections --shared-memory --max-memory=131072 --no-entry -o %t.wasm %t.o
; RUN: obj2yaml %t.wasm | FileCheck %s

; CHECK:      - Type:            GLOBAL
; CHECK-NEXT:   Globals:
; CHECK-NEXT:     - Index:           0
; CHECK-NEXT:       Type:            I32
; CHECK-NEXT:       Mutable:         true
; CHECK-NEXT:       InitExpr:
; CHECK-NEXT:         Opcode:          I32_CONST
; CHECK-NEXT:         Value:           66592

; __tls_base
; CHECK-NEXT:     - Index:           1
; CHECK-NEXT:       Type:            I32
; CHECK-NEXT:       Mutable:         true
; CHECK-NEXT:       InitExpr:
; CHECK-NEXT:         Opcode:          I32_CONST
; CHECK-NEXT:         Value:           0

; __tls_size
; CHECK-NEXT:     - Index:           2
; CHECK-NEXT:       Type:            I32
; CHECK-NEXT:       Mutable:         false
; CHECK-NEXT:       InitExpr:
; CHECK-NEXT:         Opcode:          I32_CONST
; CHECK-NEXT:         Value:           20

; __tls_align
; CHECK-NEXT:     - Index:           3
; CHECK-NEXT:       Type:            I32
; CHECK-NEXT:       Mutable:         false
; CHECK-NEXT:       InitExpr:
; CHECK-NEXT:         Opcode:          I32_CONST
; CHECK-NEXT:         Value:           16
