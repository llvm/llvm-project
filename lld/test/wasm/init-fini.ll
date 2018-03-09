; RUN: llc -filetype=obj -o %t.o %s
; RUN: llc -filetype=obj %S/Inputs/global-ctor-dtor.ll -o %t.global-ctor-dtor.o

target triple = "wasm32-unknown-unknown-wasm"

define hidden void @func1() {
entry:
  ret void
}

define hidden void @func2() {
entry:
  ret void
}

define hidden void @func3() {
entry:
  ret void
}

define hidden void @func4() {
entry:
  ret void
}

define void @__cxa_atexit() {
  ret void
}

define hidden void @_start() {
entry:
  ret void
}

@llvm.global_ctors = appending global [3 x { i32, void ()*, i8* }] [
  { i32, void ()*, i8* } { i32 1001, void ()* @func1, i8* null },
  { i32, void ()*, i8* } { i32 101, void ()* @func1, i8* null },
  { i32, void ()*, i8* } { i32 101, void ()* @func2, i8* null }
]

@llvm.global_dtors = appending global [3 x { i32, void ()*, i8* }] [
  { i32, void ()*, i8* } { i32 1001, void ()* @func3, i8* null },
  { i32, void ()*, i8* } { i32 101, void ()* @func3, i8* null },
  { i32, void ()*, i8* } { i32 101, void ()* @func4, i8* null }
]

; RUN: lld -flavor wasm --check-signatures %t.o %t.global-ctor-dtor.o -o %t.wasm
; RUN: obj2yaml %t.wasm | FileCheck %s

; CHECK:        - Type:            ELEM
; CHECK-NEXT:     Segments:
; CHECK-NEXT:       - Offset:
; CHECK-NEXT:           Opcode:          I32_CONST
; CHECK-NEXT:           Value:           1
; CHECK-NEXT:         Functions:       [ 6, 9, 13, 15, 17 ]

; CHECK:              Body:            100010011007100B100E100B10101000100A100B10120B
; CHECK-NEXT:   - Type:            CUSTOM
; CHECK-NEXT:     Name:            linking
; CHECK-NEXT:     DataSize:        0
; CHECK-NEXT:   - Type:            CUSTOM
; CHECK-NEXT:     Name:            name
; CHECK-NEXT:     FunctionNames:
; CHECK-NEXT:       - Index:           0
; CHECK-NEXT:         Name:            func1
; CHECK-NEXT:       - Index:           1
; CHECK-NEXT:         Name:            func2
; CHECK-NEXT:       - Index:           2
; CHECK-NEXT:         Name:            func3
; CHECK-NEXT:       - Index:           3
; CHECK-NEXT:         Name:            func4
; CHECK-NEXT:       - Index:           4
; CHECK-NEXT:         Name:            __cxa_atexit
; CHECK-NEXT:       - Index:           5
; CHECK-NEXT:         Name:            _start
; CHECK-NEXT:       - Index:           6
; CHECK-NEXT:         Name:            .Lcall_dtors.101
; CHECK-NEXT:       - Index:           7
; CHECK-NEXT:         Name:            .Lregister_call_dtors.101
; CHECK-NEXT:       - Index:           8
; CHECK-NEXT:         Name:            .Lbitcast
; CHECK-NEXT:       - Index:           9
; CHECK-NEXT:         Name:            .Lcall_dtors.1001
; CHECK-NEXT:       - Index:           10
; CHECK-NEXT:         Name:            .Lregister_call_dtors.1001
; CHECK-NEXT:       - Index:           11
; CHECK-NEXT:         Name:            myctor
; CHECK-NEXT:       - Index:           12
; CHECK-NEXT:         Name:            mydtor
; CHECK-NEXT:       - Index:           13
; CHECK-NEXT:         Name:            .Lcall_dtors.101
; CHECK-NEXT:       - Index:           14
; CHECK-NEXT:         Name:            .Lregister_call_dtors.101
; CHECK-NEXT:       - Index:           15
; CHECK-NEXT:         Name:            .Lcall_dtors.202
; CHECK-NEXT:       - Index:           16
; CHECK-NEXT:         Name:            .Lregister_call_dtors.202
; CHECK-NEXT:       - Index:           17
; CHECK-NEXT:         Name:            .Lcall_dtors.2002
; CHECK-NEXT:       - Index:           18
; CHECK-NEXT:         Name:            .Lregister_call_dtors.2002
; CHECK-NEXT:       - Index:           19
; CHECK-NEXT:         Name:            __wasm_call_ctors
; CHECK-NEXT: ...


; RUN: lld -flavor wasm --check-signatures -r %t.o %t.global-ctor-dtor.o -o %t.reloc.wasm
; RUN: obj2yaml %t.reloc.wasm | FileCheck -check-prefix=RELOC %s

; RELOC:          Name:            linking
; RELOC-NEXT:     DataSize:        0
; RELOC-NEXT:     SymbolInfo:
; RELOC-NEXT:       - Name:            __dso_handle
; RELOC-NEXT:         Flags:           [ BINDING_WEAK, VISIBILITY_HIDDEN ]
; RELOC-NEXT:       - Name:            func1
; RELOC-NEXT:         Flags:           [ VISIBILITY_HIDDEN ]
; RELOC-NEXT:       - Name:            func2
; RELOC-NEXT:         Flags:           [ VISIBILITY_HIDDEN ]
; RELOC-NEXT:       - Name:            func3
; RELOC-NEXT:         Flags:           [ VISIBILITY_HIDDEN ]
; RELOC-NEXT:       - Name:            func4
; RELOC-NEXT:         Flags:           [ VISIBILITY_HIDDEN ]
; RELOC-NEXT:       - Name:            _start
; RELOC-NEXT:         Flags:           [ VISIBILITY_HIDDEN ]
; RELOC-NEXT:       - Name:            .Lcall_dtors.101
; RELOC-NEXT:         Flags:           [ BINDING_LOCAL ]
; RELOC-NEXT:       - Name:            .Lregister_call_dtors.101
; RELOC-NEXT:         Flags:           [ BINDING_LOCAL ]
; RELOC-NEXT:       - Name:            .Lbitcast
; RELOC-NEXT:         Flags:           [ BINDING_LOCAL ]
; RELOC-NEXT:       - Name:            .Lcall_dtors.1001
; RELOC-NEXT:         Flags:           [ BINDING_LOCAL ]
; RELOC-NEXT:       - Name:            .Lregister_call_dtors.1001
; RELOC-NEXT:         Flags:           [ BINDING_LOCAL ]
; RELOC-NEXT:       - Name:            myctor
; RELOC-NEXT:         Flags:           [ VISIBILITY_HIDDEN ]
; RELOC-NEXT:       - Name:            mydtor
; RELOC-NEXT:         Flags:           [ VISIBILITY_HIDDEN ]
; RELOC-NEXT:       - Name:            .Lcall_dtors.101.1
; RELOC-NEXT:         Flags:           [ BINDING_LOCAL ]
; RELOC-NEXT:       - Name:            .Lregister_call_dtors.101.1
; RELOC-NEXT:         Flags:           [ BINDING_LOCAL ]
; RELOC-NEXT:       - Name:            .Lcall_dtors.202
; RELOC-NEXT:         Flags:           [ BINDING_LOCAL ]
; RELOC-NEXT:       - Name:            .Lregister_call_dtors.202
; RELOC-NEXT:         Flags:           [ BINDING_LOCAL ]
; RELOC-NEXT:       - Name:            .Lcall_dtors.2002
; RELOC-NEXT:         Flags:           [ BINDING_LOCAL ]
; RELOC-NEXT:       - Name:            .Lregister_call_dtors.2002
; RELOC-NEXT:         Flags:           [ BINDING_LOCAL ]
; RELOC-NEXT:     InitFunctions:
; RELOC-NEXT:       - Priority:        101
; RELOC-NEXT:         FunctionIndex:   0
; RELOC-NEXT:       - Priority:        101
; RELOC-NEXT:         FunctionIndex:   1
; RELOC-NEXT:       - Priority:        101
; RELOC-NEXT:         FunctionIndex:   7
; RELOC-NEXT:       - Priority:        101
; RELOC-NEXT:         FunctionIndex:   11
; RELOC-NEXT:       - Priority:        101
; RELOC-NEXT:         FunctionIndex:   14
; RELOC-NEXT:       - Priority:        202
; RELOC-NEXT:         FunctionIndex:   11
; RELOC-NEXT:       - Priority:        202
; RELOC-NEXT:         FunctionIndex:   16
; RELOC-NEXT:       - Priority:        1001
; RELOC-NEXT:         FunctionIndex:   0
; RELOC-NEXT:       - Priority:        1001
; RELOC-NEXT:         FunctionIndex:   10
; RELOC-NEXT:       - Priority:        2002
; RELOC-NEXT:         FunctionIndex:   11
; RELOC-NEXT:       - Priority:        2002
; RELOC-NEXT:         FunctionIndex:   18
; RELOC-NEXT:   - Type:            CUSTOM
; RELOC-NEXT:     Name:            name
; RELOC-NEXT:     FunctionNames:
; RELOC-NEXT:       - Index:           0
; RELOC-NEXT:         Name:            func1
; RELOC-NEXT:       - Index:           1
; RELOC-NEXT:         Name:            func2
; RELOC-NEXT:       - Index:           2
; RELOC-NEXT:         Name:            func3
; RELOC-NEXT:       - Index:           3
; RELOC-NEXT:         Name:            func4
; RELOC-NEXT:       - Index:           4
; RELOC-NEXT:         Name:            __cxa_atexit
; RELOC-NEXT:       - Index:           5
; RELOC-NEXT:         Name:            _start
; RELOC-NEXT:       - Index:           6
; RELOC-NEXT:         Name:            .Lcall_dtors.101
; RELOC-NEXT:       - Index:           7
; RELOC-NEXT:         Name:            .Lregister_call_dtors.101
; RELOC-NEXT:       - Index:           8
; RELOC-NEXT:         Name:            .Lbitcast
; RELOC-NEXT:       - Index:           9
; RELOC-NEXT:         Name:            .Lcall_dtors.1001
; RELOC-NEXT:       - Index:           10
; RELOC-NEXT:         Name:            .Lregister_call_dtors.1001
; RELOC-NEXT:       - Index:           11
; RELOC-NEXT:         Name:            myctor
; RELOC-NEXT:       - Index:           12
; RELOC-NEXT:         Name:            mydtor
; RELOC-NEXT:       - Index:           13
; RELOC-NEXT:         Name:            .Lcall_dtors.101
; RELOC-NEXT:       - Index:           14
; RELOC-NEXT:         Name:            .Lregister_call_dtors.101
; RELOC-NEXT:       - Index:           15
; RELOC-NEXT:         Name:            .Lcall_dtors.202
; RELOC-NEXT:       - Index:           16
; RELOC-NEXT:         Name:            .Lregister_call_dtors.202
; RELOC-NEXT:       - Index:           17
; RELOC-NEXT:         Name:            .Lcall_dtors.2002
; RELOC-NEXT:       - Index:           18
; RELOC-NEXT:         Name:            .Lregister_call_dtors.2002
; RELOC-NEXT: ...
