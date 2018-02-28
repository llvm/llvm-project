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

define i32 @__cxa_atexit(i32 %func, i32 %arg, i32 %dso_handle) {
  ret i32 0
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

; RUN: wasm-ld --check-signatures %t.o %t.global-ctor-dtor.o -o %t.wasm
; RUN: obj2yaml %t.wasm | FileCheck %s

; CHECK:        - Type:            ELEM
; CHECK-NEXT:     Segments:
; CHECK-NEXT:       - Offset:
; CHECK-NEXT:           Opcode:          I32_CONST
; CHECK-NEXT:           Value:           1
; CHECK-NEXT:         Functions:       [ 6, 8, 12, 14, 16 ]

; CHECK:              Body:            100010011007100A100D100A100F10001009100A10110B
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
; CHECK-NEXT:         Name:            .Lcall_dtors.1001
; CHECK-NEXT:       - Index:           9
; CHECK-NEXT:         Name:            .Lregister_call_dtors.1001
; CHECK-NEXT:       - Index:           10
; CHECK-NEXT:         Name:            myctor
; CHECK-NEXT:       - Index:           11
; CHECK-NEXT:         Name:            mydtor
; CHECK-NEXT:       - Index:           12
; CHECK-NEXT:         Name:            .Lcall_dtors.101
; CHECK-NEXT:       - Index:           13
; CHECK-NEXT:         Name:            .Lregister_call_dtors.101
; CHECK-NEXT:       - Index:           14
; CHECK-NEXT:         Name:            .Lcall_dtors.202
; CHECK-NEXT:       - Index:           15
; CHECK-NEXT:         Name:            .Lregister_call_dtors.202
; CHECK-NEXT:       - Index:           16
; CHECK-NEXT:         Name:            .Lcall_dtors.2002
; CHECK-NEXT:       - Index:           17
; CHECK-NEXT:         Name:            .Lregister_call_dtors.2002
; CHECK-NEXT:       - Index:           18
; CHECK-NEXT:         Name:            __wasm_call_ctors
; CHECK-NEXT: ...


; RUN: wasm-ld --check-signatures -r %t.o %t.global-ctor-dtor.o -o %t.reloc.wasm
; RUN: obj2yaml %t.reloc.wasm | FileCheck -check-prefix=RELOC %s

; RELOC:          Name:            linking
; RELOC-NEXT:     SymbolTable:
; RELOC-NEXT:       - Index:           0
; RELOC-NEXT:         Kind:            FUNCTION
; RELOC-NEXT:         Name:            func1
; RELOC-NEXT:         Flags:           [ VISIBILITY_HIDDEN ]
; RELOC-NEXT:         Function:        0
; RELOC-NEXT:       - Index:           1
; RELOC-NEXT:         Kind:            FUNCTION
; RELOC-NEXT:         Name:            func2
; RELOC-NEXT:         Flags:           [ VISIBILITY_HIDDEN ]
; RELOC-NEXT:         Function:        1
; RELOC-NEXT:       - Index:           2
; RELOC-NEXT:         Kind:            FUNCTION
; RELOC-NEXT:         Name:            func3
; RELOC-NEXT:         Flags:           [ VISIBILITY_HIDDEN ]
; RELOC-NEXT:         Function:        2
; RELOC-NEXT:       - Index:           3
; RELOC-NEXT:         Kind:            FUNCTION
; RELOC-NEXT:         Name:            func4
; RELOC-NEXT:         Flags:           [ VISIBILITY_HIDDEN ]
; RELOC-NEXT:         Function:        3
; RELOC-NEXT:       - Index:           4
; RELOC-NEXT:         Kind:            FUNCTION
; RELOC-NEXT:         Name:            __cxa_atexit
; RELOC-NEXT:         Flags:           [  ]
; RELOC-NEXT:         Function:        4
; RELOC-NEXT:       - Index:           5
; RELOC-NEXT:         Kind:            FUNCTION
; RELOC-NEXT:         Name:            _start
; RELOC-NEXT:         Flags:           [ VISIBILITY_HIDDEN ]
; RELOC-NEXT:         Function:        5
; RELOC-NEXT:       - Index:           6
; RELOC-NEXT:         Kind:            FUNCTION
; RELOC-NEXT:         Name:            .Lcall_dtors.101
; RELOC-NEXT:         Flags:           [ BINDING_LOCAL ]
; RELOC-NEXT:         Function:        6
; RELOC-NEXT:       - Index:           7
; RELOC-NEXT:         Kind:            FUNCTION
; RELOC-NEXT:         Name:            .Lregister_call_dtors.101
; RELOC-NEXT:         Flags:           [ BINDING_LOCAL ]
; RELOC-NEXT:         Function:        7
; RELOC-NEXT:       - Index:           8
; RELOC-NEXT:         Kind:            DATA
; RELOC-NEXT:         Name:            __dso_handle
; RELOC-NEXT:         Flags:           [ BINDING_WEAK, VISIBILITY_HIDDEN, UNDEFINED ]
; RELOC-NEXT:       - Index:           9
; RELOC-NEXT:         Kind:            FUNCTION
; RELOC-NEXT:         Name:            .Lcall_dtors.1001
; RELOC-NEXT:         Flags:           [ BINDING_LOCAL ]
; RELOC-NEXT:         Function:        8
; RELOC-NEXT:       - Index:           10
; RELOC-NEXT:         Kind:            FUNCTION
; RELOC-NEXT:         Name:            .Lregister_call_dtors.1001
; RELOC-NEXT:         Flags:           [ BINDING_LOCAL ]
; RELOC-NEXT:         Function:        9
; RELOC-NEXT:       - Index:           11
; RELOC-NEXT:         Kind:            FUNCTION
; RELOC-NEXT:         Name:            myctor
; RELOC-NEXT:         Flags:           [ VISIBILITY_HIDDEN ]
; RELOC-NEXT:         Function:        10
; RELOC-NEXT:       - Index:           12
; RELOC-NEXT:         Kind:            FUNCTION
; RELOC-NEXT:         Name:            mydtor
; RELOC-NEXT:         Flags:           [ VISIBILITY_HIDDEN ]
; RELOC-NEXT:         Function:        11
; RELOC-NEXT:       - Index:           13
; RELOC-NEXT:         Kind:            GLOBAL
; RELOC-NEXT:         Name:            __stack_pointer
; RELOC-NEXT:         Flags:           [ UNDEFINED ]
; RELOC-NEXT:         Global:          0
; RELOC-NEXT:       - Index:           14
; RELOC-NEXT:         Kind:            FUNCTION
; RELOC-NEXT:         Name:            .Lcall_dtors.101
; RELOC-NEXT:         Flags:           [ BINDING_LOCAL ]
; RELOC-NEXT:         Function:        12
; RELOC-NEXT:       - Index:           15
; RELOC-NEXT:         Kind:            FUNCTION
; RELOC-NEXT:         Name:            .Lregister_call_dtors.101
; RELOC-NEXT:         Flags:           [ BINDING_LOCAL ]
; RELOC-NEXT:         Function:        13
; RELOC-NEXT:       - Index:           16
; RELOC-NEXT:         Kind:            FUNCTION
; RELOC-NEXT:         Name:            .Lcall_dtors.202
; RELOC-NEXT:         Flags:           [ BINDING_LOCAL ]
; RELOC-NEXT:         Function:        14
; RELOC-NEXT:       - Index:           17
; RELOC-NEXT:         Kind:            FUNCTION
; RELOC-NEXT:         Name:            .Lregister_call_dtors.202
; RELOC-NEXT:         Flags:           [ BINDING_LOCAL ]
; RELOC-NEXT:         Function:        15
; RELOC-NEXT:       - Index:           18
; RELOC-NEXT:         Kind:            FUNCTION
; RELOC-NEXT:         Name:            .Lcall_dtors.2002
; RELOC-NEXT:         Flags:           [ BINDING_LOCAL ]
; RELOC-NEXT:         Function:        16
; RELOC-NEXT:       - Index:           19
; RELOC-NEXT:         Kind:            FUNCTION
; RELOC-NEXT:         Name:            .Lregister_call_dtors.2002
; RELOC-NEXT:         Flags:           [ BINDING_LOCAL ]
; RELOC-NEXT:         Function:        17
; RELOC-NEXT:     InitFunctions:
; RELOC-NEXT:       - Priority:        101
; RELOC-NEXT:         Symbol:          0
; RELOC-NEXT:       - Priority:        101
; RELOC-NEXT:         Symbol:          1
; RELOC-NEXT:       - Priority:        101
; RELOC-NEXT:         Symbol:          7
; RELOC-NEXT:       - Priority:        101
; RELOC-NEXT:         Symbol:          11
; RELOC-NEXT:       - Priority:        101
; RELOC-NEXT:         Symbol:          15
; RELOC-NEXT:       - Priority:        202
; RELOC-NEXT:         Symbol:          11
; RELOC-NEXT:       - Priority:        202
; RELOC-NEXT:         Symbol:          17
; RELOC-NEXT:       - Priority:        1001
; RELOC-NEXT:         Symbol:          0
; RELOC-NEXT:       - Priority:        1001
; RELOC-NEXT:         Symbol:          10
; RELOC-NEXT:       - Priority:        2002
; RELOC-NEXT:         Symbol:          11
; RELOC-NEXT:       - Priority:        2002
; RELOC-NEXT:         Symbol:          19
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
; RELOC-NEXT:         Name:            .Lcall_dtors.1001
; RELOC-NEXT:       - Index:           9
; RELOC-NEXT:         Name:            .Lregister_call_dtors.1001
; RELOC-NEXT:       - Index:           10
; RELOC-NEXT:         Name:            myctor
; RELOC-NEXT:       - Index:           11
; RELOC-NEXT:         Name:            mydtor
; RELOC-NEXT:       - Index:           12
; RELOC-NEXT:         Name:            .Lcall_dtors.101
; RELOC-NEXT:       - Index:           13
; RELOC-NEXT:         Name:            .Lregister_call_dtors.101
; RELOC-NEXT:       - Index:           14
; RELOC-NEXT:         Name:            .Lcall_dtors.202
; RELOC-NEXT:       - Index:           15
; RELOC-NEXT:         Name:            .Lregister_call_dtors.202
; RELOC-NEXT:       - Index:           16
; RELOC-NEXT:         Name:            .Lcall_dtors.2002
; RELOC-NEXT:       - Index:           17
; RELOC-NEXT:         Name:            .Lregister_call_dtors.2002
; RELOC-NEXT: ...
