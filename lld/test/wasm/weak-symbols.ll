; RUN: llc -filetype=obj -mtriple=wasm32-unknown-unknown-wasm %p/Inputs/weak-symbol1.ll -o %t1.o
; RUN: llc -filetype=obj -mtriple=wasm32-unknown-unknown-wasm %p/Inputs/weak-symbol2.ll -o %t2.o
; RUN: llc -filetype=obj -mtriple=wasm32-unknown-unknown-wasm %s -o %t.o
; RUN: lld -flavor wasm -o %t.wasm %t.o %t1.o %t2.o
; RUN: obj2yaml %t.wasm | FileCheck %s

declare i32 @weakFn() local_unnamed_addr

define void @_start() local_unnamed_addr {
entry:
  %call = call i32 @weakFn()
  ret void
}

; CHECK:      --- !WASM
; CHECK-NEXT: FileHeader:
; CHECK-NEXT:   Version:         0x00000001
; CHECK-NEXT: Sections:
; CHECK-NEXT:   - Type:            TYPE
; CHECK-NEXT:     Signatures:
; CHECK-NEXT:       - Index:           0
; CHECK-NEXT:         ReturnType:      NORESULT
; CHECK-NEXT:         ParamTypes:
; CHECK-NEXT:       - Index:           1
; CHECK-NEXT:         ReturnType:      I32
; CHECK-NEXT:         ParamTypes:
; CHECK-NEXT:   - Type:            FUNCTION
; CHECK-NEXT:     FunctionTypes:   [ 0, 1, 1, 1, 1 ]
; CHECK-NEXT:   - Type:            TABLE
; CHECK-NEXT:     Tables:
; CHECK-NEXT:       - ElemType:        ANYFUNC
; CHECK-NEXT:         Limits:
; CHECK-NEXT:           Flags:           [ HAS_MAX ]
; CHECK-NEXT:           Initial:         0x00000002
; CHECK-NEXT:           Maximum:         0x00000002
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
; CHECK-NEXT:       - Name:            weakFn
; CHECK-NEXT:         Kind:            FUNCTION
; CHECK-NEXT:         Index:           1
; CHECK-NEXT:       - Name:            exportWeak1
; CHECK-NEXT:         Kind:            FUNCTION
; CHECK-NEXT:         Index:           2
; CHECK-NEXT:       - Name:            exportWeak2
; CHECK-NEXT:         Kind:            FUNCTION
; CHECK-NEXT:         Index:           4
; CHECK-NEXT:   - Type:            ELEM
; CHECK-NEXT:     Segments:
; CHECK-NEXT:       - Offset:
; CHECK-NEXT:           Opcode:          I32_CONST
; CHECK-NEXT:           Value:           1
; CHECK-NEXT:         Functions:       [ 1 ]
; CHECK-NEXT:   - Type:            CODE
; CHECK-NEXT:     Functions:
; CHECK-NEXT:       - Locals:
; CHECK-NEXT:         Body:            1081808080001A0B
; CHECK-NEXT:       - Locals:
; CHECK-NEXT:         Body:            41010B
; CHECK-NEXT:       - Locals:
; CHECK-NEXT:         Body:            4181808080000B
; CHECK-NEXT:       - Locals:
; CHECK-NEXT:         Body:            41020B
; CHECK-NEXT:       - Locals:
; CHECK-NEXT:         Body:            4181808080000B
; CHECK-NEXT:   - Type:            CUSTOM
; CHECK-NEXT:     Name:            linking
; CHECK-NEXT:     DataSize:        0
; CHECK-NEXT:   - Type:            CUSTOM
; CHECK-NEXT:     Name:            name
; CHECK-NEXT:     FunctionNames:
; CHECK-NEXT:       - Index:           0
; CHECK-NEXT:         Name:            _start
; CHECK-NEXT:       - Index:           2
; CHECK-NEXT:         Name:            exportWeak1
; CHECK-NEXT:       - Index:           4
; CHECK-NEXT:         Name:            exportWeak2
; CHECK-NEXT: ...
