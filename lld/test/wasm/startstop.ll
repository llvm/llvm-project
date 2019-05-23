; RUN: llc -filetype=obj -o %t.o %s
; RUN: wasm-ld --no-gc-sections %t.o -o %t.wasm
; RUN: obj2yaml %t.wasm | FileCheck %s

target triple = "wasm32-unknown-unknown"

@foo = global i32 3, section "mysection", align 4
@bar = global i32 4, section "mysection", align 4

@__start_mysection = external global i8*
@__stop_mysection = external global i8*

define i8** @get_start() {
  ret i8** @__start_mysection
}

define i8** @get_end() {
  ret i8** @__stop_mysection
}

define void @_start()  {
entry:
  ret void
}
; CHECK:        - Type:            CODE
; CHECK-NEXT:     Functions:
; CHECK-NEXT:       - Index:           0
; CHECK-NEXT:         Locals:          []
; CHECK-NEXT:         Body:            0B
; CHECK-NEXT:       - Index:           1
; CHECK-NEXT:         Locals:          []
; CHECK-NEXT:         Body:            4180888080000B
; CHECK-NEXT:       - Index:           2
; CHECK-NEXT:         Locals:          []
; CHECK-NEXT:         Body:            4188888080000B
; CHECK-NEXT:       - Index:           3
; CHECK-NEXT:         Locals:          []
; CHECK-NEXT:         Body:            0B
; CHECK-NEXT:   - Type:            DATA
; CHECK-NEXT:     Segments:
; CHECK-NEXT:       - SectionOffset:   7
; CHECK-NEXT:         InitFlags:       0
; CHECK-NEXT:         Offset:
; CHECK-NEXT:           Opcode:          I32_CONST
; CHECK-NEXT:           Value:           1024
; CHECK-NEXT:         Content:         '0300000004000000'
; CHECK-NEXT:   - Type:            CUSTOM
; CHECK-NEXT:     Name:            name
; CHECK-NEXT:     FunctionNames:
; CHECK-NEXT:       - Index:           0
; CHECK-NEXT:         Name:            __wasm_call_ctors
; CHECK-NEXT:       - Index:           1
; CHECK-NEXT:         Name:            get_start
; CHECK-NEXT:       - Index:           2
; CHECK-NEXT:         Name:            get_end
; CHECK-NEXT:       - Index:           3
; CHECK-NEXT:         Name:            _start
