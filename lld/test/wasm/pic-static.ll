; Test that PIC code can be linked into static binaries.
; In this case the GOT entries will end up as internalized wasm globals with
; fixed values.
; RUN: llvm-mc -filetype=obj -triple=wasm32-unknown-unknown %p/Inputs/ret32.s -o %t.ret32.o
; RUN: llc -relocation-model=pic -filetype=obj %s -o %t.o
; RUN: wasm-ld --allow-undefined --export-all -o %t.wasm %t.o %t.ret32.o
; RUN: obj2yaml %t.wasm | FileCheck %s

target triple = "wasm32-unknown-emscripten"

declare i32 @ret32(float)
declare i32 @missing_function(float)
@global_float = global float 1.0
@hidden_float = hidden global float 2.0
@missing_float = extern_weak global float

@ret32_ptr = global ptr @ret32, align 4

define ptr @getaddr_external() {
  ret ptr @ret32;
}

define ptr @getaddr_missing_function() {
  ret ptr @missing_function;
}

define ptr @getaddr_hidden() {
  ret ptr @hidden_func;
}

define ptr @getaddr_missing_float() {
  ret ptr @missing_float
}

define hidden i32 @hidden_func() {
  ret i32 1
}

define void @_start() {
entry:
  %f = load float, ptr @hidden_float, align 4
  %addr = load ptr, ptr @ret32_ptr, align 4
  %arg = load float, ptr @global_float, align 4
  call i32 %addr(float %arg)

  %addr2 = call ptr @getaddr_external()
  %arg2 = load float, ptr @hidden_float, align 4
  call i32 %addr2(float %arg2)

  %addr3 = call ptr @getaddr_hidden()
  call i32 %addr3()

  ret void
}

; CHECK:        - Type:            GLOBAL
; CHECK-NEXT:     Globals:

; __stack_pointer
; CHECK-NEXT:       - Index:           0
; CHECK-NEXT:         Type:            I32
; CHECK-NEXT:         Mutable:         true
; CHECK-NEXT:         InitExpr:
; CHECK-NEXT:           Opcode:          I32_CONST
; CHECK-NEXT:           Value:           66576

; GOT.func.ret32
; CHECK-NEXT:       - Index:           1
; CHECK-NEXT:         Type:            I32
; CHECK-NEXT:         Mutable:         false
; CHECK-NEXT:         InitExpr:
; CHECK-NEXT:           Opcode:          I32_CONST
; CHECK-NEXT:           Value:           1

; GOT.func.missing_function
; CHECK-NEXT:       - Index:           2
; CHECK-NEXT:         Type:            I32
; CHECK-NEXT:         Mutable:         false
; CHECK-NEXT:         InitExpr:
; CHECK-NEXT:           Opcode:          I32_CONST
; CHECK-NEXT:           Value:           2

; __table_base
; CHECK-NEXT:       - Index:           3
; CHECK-NEXT:         Type:            I32
; CHECK-NEXT:         Mutable:         false
; CHECK-NEXT:         InitExpr:
; CHECK-NEXT:           Opcode:          I32_CONST
; CHECK-NEXT:           Value:           1

; GOT.mem.missing_float
; CHECK-NEXT:       - Index:           4
; CHECK-NEXT:         Type:            I32
; CHECK-NEXT:         Mutable:         false
; CHECK-NEXT:         InitExpr:
; CHECK-NEXT:           Opcode:          I32_CONST
; CHECK-NEXT:           Value:           0

; GOT.mem.global_float
; CHECK-NEXT:       - Index:           5
; CHECK-NEXT:         Type:            I32
; CHECK-NEXT:         Mutable:         false
; CHECK-NEXT:         InitExpr:
; CHECK-NEXT:           Opcode:          I32_CONST
; CHECK-NEXT:           Value:           1024

; GOT.mem.ret32_ptr
; CHECK-NEXT:       - Index:           6
; CHECK-NEXT:         Type:            I32
; CHECK-NEXT:         Mutable:         false
; CHECK-NEXT:         InitExpr:
; CHECK-NEXT:           Opcode:          I32_CONST
; CHECK-NEXT:           Value:           1032

; __memory_base
; CHECK-NEXT:       - Index:           7
; CHECK-NEXT:         Type:            I32
; CHECK-NEXT:         Mutable:         false
; CHECK-NEXT:         InitExpr:
; CHECK-NEXT:           Opcode:          I32_CONST
; CHECK-NEXT:           Value:           0
