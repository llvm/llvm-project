; RUN: llc -mtriple wasm32-unknown-unknown-wasm -filetype=obj -o %t.o %s
; RUN: llc -mtriple=wasm32-unknown-unknown-wasm -filetype=obj %S/Inputs/weak-alias.ll -o %t2.o
; RUN: lld -flavor wasm %t.o %t2.o -o %t.wasm
; RUN: obj2yaml %t.wasm | FileCheck %s

; Test that weak aliases (bar is a weak alias of foo) are linked correctly

declare i32 @bar() local_unnamed_addr #1

; Function Attrs: nounwind uwtable
define i32 @_start() local_unnamed_addr #1 {
entry:
  %call = tail call i32 @bar() #2
  ret i32 %call
}

; CHECK:      --- !WASM
; CHECK-NEXT: FileHeader:
; CHECK-NEXT:   Version:         0x00000001
; CHECK-NEXT: Sections:
; CHECK-NEXT:   - Type:            TYPE
; CHECK-NEXT:     Signatures:
; CHECK-NEXT:       - Index:           0
; CHECK-NEXT:         ReturnType:      I32
; CHECK-NEXT:         ParamTypes:
; CHECK-NEXT:   - Type:            FUNCTION
; CHECK-NEXT:     FunctionTypes:   [ 0, 0, 0 ]
; CHECK-NEXT:   - Type:            TABLE
; CHECK-NEXT:     Tables:
; CHECK-NEXT:       - ElemType:        ANYFUNC
; CHECK-NEXT:         Limits:
; CHECK-NEXT:           Flags:           0x00000001
; CHECK-NEXT:           Initial:         0x00000001
; CHECK-NEXT:           Maximum:         0x00000001
; CHECK-NEXT:   - Type:            MEMORY
; CHECK-NEXT:     Memories:
; CHECK-NEXT:       - Initial:         0x00000002
; CHECK-NEXT:   - Type:            GLOBAL
; CHECK-NEXT:     Globals:
; CHECK-NEXT:       - Type:            I32
; CHECK-NEXT:         Mutable:         true
; CHECK-NEXT:         InitExpr:
; CHECK-NEXT:           Opcode:          I32_CONST
; CHECK-NEXT:           Value:           66560
; CHECK-NEXT:   - Type:            EXPORT
; CHECK-NEXT:     Exports:
; CHECK-NEXT:       - Name:            memory
; CHECK-NEXT:         Kind:            MEMORY
; CHECK-NEXT:         Index:           0
; CHECK-NEXT:       - Name:            _start
; CHECK-NEXT:         Kind:            FUNCTION
; CHECK-NEXT:         Index:           0
; CHECK-NEXT:       - Name:            bar
; CHECK-NEXT:         Kind:            FUNCTION
; CHECK-NEXT:         Index:           1
; CHECK-NEXT:       - Name:            foo
; CHECK-NEXT:         Kind:            FUNCTION
; CHECK-NEXT:         Index:           1
; CHECK-NEXT:       - Name:            call_bar
; CHECK-NEXT:         Kind:            FUNCTION
; CHECK-NEXT:         Index:           2
; CHECK-NEXT:   - Type:            CODE
; CHECK-NEXT:     Functions:
; CHECK-NEXT:       - Locals:
; CHECK-NEXT:         Body:            1081808080000B
; CHECK-NEXT:       - Locals:
; CHECK-NEXT:         Body:            41000B
; CHECK-NEXT:       - Locals:
; CHECK-NEXT:         Body:            1081808080000B
; CHECK-NEXT:   - Type:            CUSTOM
; CHECK-NEXT:     Name:            linking
; CHECK-NEXT:     DataSize:        0
; CHECK-NEXT:   - Type:            CUSTOM
; CHECK-NEXT:     Name:            name
; CHECK-NEXT:     FunctionNames:
; CHECK-NEXT:       - Index:           0
; CHECK-NEXT:         Name:            _start
; CHECK-NEXT:       - Index:           1
; CHECK-NEXT:         Name:            foo
; CHECK-NEXT:       - Index:           2
; CHECK-NEXT:         Name:            call_bar
; CHECK-NEXT: ...
