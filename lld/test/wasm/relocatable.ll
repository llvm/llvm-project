; RUN: llvm-mc -filetype=obj -triple=wasm32-unknown-unknown %p/Inputs/hello.s -o %t.hello.o
; RUN: llc -filetype=obj %s -o %t.o
; RUN: wasm-ld -r -o %t2.o %t.hello.o %t.o
; RUN: obj2yaml %t2.o | FileCheck %s

; Verify the resulting object can be used as linker input
; RUN: wasm-ld --allow-undefined -o %t.wasm %t2.o --export-table

target triple = "wasm32-unknown-unknown"

; Function Attrs: nounwind
define hidden i32 @my_func() local_unnamed_addr {
entry:
  %call = tail call i32 @foo_import()
  %call2 = tail call i32 @bar_import()
  ret i32 1
}

declare i32 @foo_import() local_unnamed_addr
declare extern_weak i32 @bar_import() local_unnamed_addr
@data_import = external global i64

@func_addr1 = hidden global i32()* @my_func, align 4
@func_addr2 = hidden global i32()* @foo_import, align 4
@func_addr3 = hidden global i32()* @bar_import, align 4
@data_addr1 = hidden global i64* @data_import, align 8

$func_comdat = comdat any
@data_comdat = weak_odr constant [3 x i8] c"abc", comdat($func_comdat)
define linkonce_odr i32 @func_comdat() comdat {
entry:
  ret i32 ptrtoint ([3 x i8]* @data_comdat to i32)
}

; Test that __attribute__(used) (i.e NO_STRIP) is preserved in the relocated symbol table
@llvm.used = appending global [1 x i8*] [i8* bitcast (i32 ()* @my_func to i8*)], section "llvm.metadata"

define void @_start() {
  ret void
}


; CHECK:      --- !WASM
; CHECK-NEXT: FileHeader:
; CHECK-NEXT:   Version:         0x1
; CHECK-NEXT: Sections:
; CHECK-NEXT:   - Type:            TYPE
; CHECK-NEXT:     Signatures:
; CHECK-NEXT:       - Index:           0
; CHECK-NEXT:         ParamTypes:
; CHECK-NEXT:           - I32
; CHECK-NEXT:         ReturnTypes:     []
; CHECK-NEXT:       - Index:           1
; CHECK-NEXT:         ParamTypes:
; CHECK-NEXT:         ReturnTypes:
; CHECK-NEXT:           - I32
; CHECK-NEXT:       - Index:           2
; CHECK-NEXT:         ParamTypes:
; CHECK-NEXT:         ReturnTypes:     []
; CHECK-NEXT:   - Type:            IMPORT
; CHECK-NEXT:     Imports:
; CHECK-NEXT:       - Module:          env
; CHECK-NEXT:         Field:           __indirect_function_table
; CHECK-NEXT:         Kind:            TABLE
; CHECK-NEXT:         Table:
; CHECK-NEXT:           Index:           0
; CHECK-NEXT:           ElemType:        FUNCREF
; CHECK-NEXT:           Limits:
; CHECK-NEXT:             Initial:         0x4
; CHECK-NEXT:       - Module:          env
; CHECK-NEXT:         Field:           puts
; CHECK-NEXT:         Kind:            FUNCTION
; CHECK-NEXT:         SigIndex:        0
; CHECK-NEXT:       - Module:          env
; CHECK-NEXT:         Field:           foo_import
; CHECK-NEXT:         Kind:            FUNCTION
; CHECK-NEXT:         SigIndex:        1
; CHECK-NEXT:       - Module:          env
; CHECK-NEXT:         Field:           bar_import
; CHECK-NEXT:         Kind:            FUNCTION
; CHECK-NEXT:         SigIndex:        1
; CHECK-NEXT:   - Type:            FUNCTION
; CHECK-NEXT:     FunctionTypes:   [ 2, 1, 1, 2 ]
; CHECK-NEXT:   - Type:            MEMORY
; CHECK-NEXT:     Memories:
; CHECK-NEXT:      - Initial:         0x1
; CHECK-NEXT:   - Type:            ELEM
; CHECK-NEXT:     Segments:
; CHECK-NEXT:       - Offset:
; CHECK-NEXT:           Opcode:          I32_CONST
; CHECK-NEXT:           Value:           1
; CHECK-NEXT:         Functions:       [ 4, 1, 2 ]
; SHARED-NEXT:  - Type:            DATACOUNT
; SHARED-NEXT:    Count:           6
; CHECK-NEXT:   - Type:            CODE
; CHECK-NEXT:     Relocations:
; CHECK-NEXT:       - Type:            R_WASM_MEMORY_ADDR_SLEB
; CHECK-NEXT:         Index:           1
; CHECK-NEXT:         Offset:          0x4
; CHECK-NEXT:       - Type:            R_WASM_FUNCTION_INDEX_LEB
; CHECK-NEXT:         Index:           2
; CHECK-NEXT:         Offset:          0xA
; CHECK-NEXT:       - Type:            R_WASM_FUNCTION_INDEX_LEB
; CHECK-NEXT:         Index:           4
; CHECK-NEXT:         Offset:          0x13
; CHECK-NEXT:       - Type:            R_WASM_FUNCTION_INDEX_LEB
; CHECK-NEXT:         Index:           5
; CHECK-NEXT:         Offset:          0x1A
; CHECK-NEXT:       - Type:            R_WASM_MEMORY_ADDR_SLEB
; CHECK-NEXT:         Index:           7
; CHECK-NEXT:         Offset:          0x26
; CHECK-NEXT:     Functions:
; CHECK-NEXT:       - Index:         3
; CHECK-NEXT:         Locals:
; CHECK-NEXT:         Body:          4180808080001080808080000B
; CHECK-NEXT:       - Index:         4
; CHECK-NEXT:         Locals:
; CHECK-NEXT:         Body:          1081808080001A1082808080001A41010B
; CHECK-NEXT:       - Index:         5
; CHECK-NEXT:         Locals:
; CHECK-NEXT:         Body:          4187808080000B
; NORMAL-NEXT:  - Type:            DATA
; NORMAL-NEXT:    Relocations:
; NORMAL-NEXT:      - Type:            R_WASM_TABLE_INDEX_I32
; NORMAL-NEXT:        Index:           3
; NORMAL-NEXT:        Offset:          0x1A
; NORMAL-NEXT:      - Type:            R_WASM_TABLE_INDEX_I32
; NORMAL-NEXT:        Index:           4
; NORMAL-NEXT:        Offset:          0x23
; NORMAL-NEXT:      - Type:            R_WASM_TABLE_INDEX_I32
; NORMAL-NEXT:        Index:           5
; NORMAL-NEXT:        Offset:          0x2C
; NORMAL-NEXT:      - Type:            R_WASM_MEMORY_ADDR_I32
; NORMAL-NEXT:        Index:           12
; NORMAL-NEXT:        Offset:          0x35
; NORMAL-NEXT:    Segments:
; NORMAL-NEXT:      - SectionOffset:   6
; NORMAL-NEXT:        InitFlags:       0
; NORMAL-NEXT:        Offset:
; NORMAL-NEXT:          Opcode:          I32_CONST
; NORMAL-NEXT:          Value:           0
; NORMAL-NEXT:        Content:         68656C6C6F0A00
; NORMAL-NEXT:      - SectionOffset:   18
; NORMAL-NEXT:        InitFlags:       0
; NORMAL-NEXT:        Offset:
; NORMAL-NEXT:          Opcode:          I32_CONST
; NORMAL-NEXT:          Value:           7
; NORMAL-NEXT:        Content:         '616263'
; NORMAL-NEXT:      - SectionOffset:   26
; NORMAL-NEXT:        InitFlags:       0
; NORMAL-NEXT:        Offset:
; NORMAL-NEXT:          Opcode:          I32_CONST
; NORMAL-NEXT:          Value:           12
; NORMAL-NEXT:        Content:         '01000000'
; NORMAL-NEXT:      - SectionOffset:   35
; NORMAL-NEXT:        InitFlags:       0
; NORMAL-NEXT:        Offset:
; NORMAL-NEXT:          Opcode:          I32_CONST
; NORMAL-NEXT:          Value:           16
; NORMAL-NEXT:        Content:         '02000000'
; NORMAL-NEXT:      - SectionOffset:   44
; NORMAL-NEXT:        InitFlags:       0
; NORMAL-NEXT:        Offset:
; NORMAL-NEXT:          Opcode:          I32_CONST
; NORMAL-NEXT:          Value:           20
; NORMAL-NEXT:        Content:         '03000000'
; NORMAL-NEXT:      - SectionOffset:   53
; NORMAL-NEXT:        InitFlags:       0
; NORMAL-NEXT:        Offset:
; NORMAL-NEXT:          Opcode:          I32_CONST
; NORMAL-NEXT:          Value:           24
; NORMAL-NEXT:        Content:         '00000000'
; NORMAL-NEXT:  - Type:            CUSTOM
; NORMAL-NEXT:    Name:            linking
; NORMAL-NEXT:    Version:         2
; NORMAL-NEXT:    SymbolTable:
; NORMAL-NEXT:      - Index:           0
; NORMAL-NEXT:        Kind:            FUNCTION
; NORMAL-NEXT:        Name:            hello
; NORMAL-NEXT:        Flags:           [ VISIBILITY_HIDDEN ]
; NORMAL-NEXT:        Function:        3
; NORMAL-NEXT:      - Index:           1
; NORMAL-NEXT:        Kind:            DATA
; NORMAL-NEXT:        Name:            hello_str
; NORMAL-NEXT:        Flags:           [  ]
; NORMAL-NEXT:        Segment:         0
; NORMAL-NEXT:        Size:            7
; NORMAL-NEXT:      - Index:           2
; NORMAL-NEXT:        Kind:            FUNCTION
; NORMAL-NEXT:        Name:            puts
; NORMAL-NEXT:        Flags:           [ UNDEFINED ]
; NORMAL-NEXT:        Function:        0
; NORMAL-NEXT:      - Index:           3
; NORMAL-NEXT:        Kind:            FUNCTION
; NORMAL-NEXT:        Name:            my_func
; NORMAL-NEXT:        Flags:           [ VISIBILITY_HIDDEN, NO_STRIP ]
; NORMAL-NEXT:        Function:        4
; NORMAL-NEXT:      - Index:           4
; NORMAL-NEXT:        Kind:            FUNCTION
; NORMAL-NEXT:        Name:            foo_import
; NORMAL-NEXT:        Flags:           [ UNDEFINED ]
; NORMAL-NEXT:        Function:        1
; NORMAL-NEXT:      - Index:           5
; NORMAL-NEXT:        Kind:            FUNCTION
; NORMAL-NEXT:        Name:            bar_import
; NORMAL-NEXT:        Flags:           [ BINDING_WEAK, UNDEFINED ]
; NORMAL-NEXT:        Function:        2
; NORMAL-NEXT:      - Index:           6
; NORMAL-NEXT:        Kind:            FUNCTION
; NORMAL-NEXT:        Name:            func_comdat
; NORMAL-NEXT:        Flags:           [ BINDING_WEAK ]
; NORMAL-NEXT:        Function:        5
; NORMAL-NEXT:      - Index:           7
; NORMAL-NEXT:        Kind:            DATA
; NORMAL-NEXT:        Name:            data_comdat
; NORMAL-NEXT:        Flags:           [ BINDING_WEAK ]
; NORMAL-NEXT:        Segment:         1
; NORMAL-NEXT:        Size:            3
; NORMAL-NEXT:      - Index:           8
; NORMAL-NEXT:        Kind:            DATA
; NORMAL-NEXT:        Name:            func_addr1
; NORMAL-NEXT:        Flags:           [ VISIBILITY_HIDDEN ]
; NORMAL-NEXT:        Segment:         2
; NORMAL-NEXT:        Size:            4
; NORMAL-NEXT:      - Index:           9
; NORMAL-NEXT:        Kind:            DATA
; NORMAL-NEXT:        Name:            func_addr2
; NORMAL-NEXT:        Flags:           [ VISIBILITY_HIDDEN ]
; NORMAL-NEXT:        Segment:         3
; NORMAL-NEXT:        Size:            4
; NORMAL-NEXT:      - Index:           10
; NORMAL-NEXT:        Kind:            DATA
; NORMAL-NEXT:        Name:            func_addr3
; NORMAL-NEXT:        Flags:           [ VISIBILITY_HIDDEN ]
; NORMAL-NEXT:        Segment:         4
; NORMAL-NEXT:        Size:            4
; NORMAL-NEXT:      - Index:           11
; NORMAL-NEXT:        Kind:            DATA
; NORMAL-NEXT:        Name:            data_addr1
; NORMAL-NEXT:        Flags:           [ VISIBILITY_HIDDEN ]
; NORMAL-NEXT:        Segment:         5
; NORMAL-NEXT:        Size:            4
; NORMAL-NEXT:      - Index:           12
; NORMAL-NEXT:        Kind:            DATA
; NORMAL-NEXT:        Name:            data_import
; NORMAL-NEXT:        Flags:           [ UNDEFINED ]
; NORMAL-NEXT:    SegmentInfo:
; NORMAL-NEXT:      - Index:           0
; NORMAL-NEXT:        Name:            .rodata.hello_str
; NORMAL-NEXT:        Alignment:       0
; NORMAL-NEXT:        Flags:           [  ]
; NORMAL-NEXT:      - Index:           1
; NORMAL-NEXT:        Name:            .rodata.data_comdat
; NORMAL-NEXT:        Alignment:       0
; NORMAL-NEXT:        Flags:           [  ]
; NORMAL-NEXT:      - Index:           2
; NORMAL-NEXT:        Name:            .data.func_addr1
; NORMAL-NEXT:        Alignment:       2
; NORMAL-NEXT:        Flags:           [  ]
; NORMAL-NEXT:      - Index:           3
; NORMAL-NEXT:        Name:            .data.func_addr2
; NORMAL-NEXT:        Alignment:       2
; NORMAL-NEXT:        Flags:           [  ]
; NORMAL-NEXT:      - Index:           4
; NORMAL-NEXT:        Name:            .data.func_addr3
; NORMAL-NEXT:        Alignment:       2
; NORMAL-NEXT:        Flags:           [  ]
; NORMAL-NEXT:      - Index:           5
; NORMAL-NEXT:        Name:            .data.data_addr1
; NORMAL-NEXT:        Alignment:       3
; NORMAL-NEXT:        Flags:           [  ]
; NORMAL-NEXT:    Comdats:
; NORMAL-NEXT:      - Name:            func_comdat
; NORMAL-NEXT:        Entries:
; NORMAL-NEXT:          - Kind:            FUNCTION
; NORMAL-NEXT:            Index:           5
; NORMAL-NEXT:          - Kind:            DATA
; NORMAL-NEXT:            Index:           1
; NORMAL-NEXT:  - Type:            CUSTOM
; NORMAL-NEXT:    Name:            name
; NORMAL-NEXT:    FunctionNames:
; NORMAL-NEXT:      - Index:           0
; NORMAL-NEXT:        Name:            puts
; NORMAL-NEXT:      - Index:           1
; NORMAL-NEXT:        Name:            foo_import
; NORMAL-NEXT:      - Index:           2
; NORMAL-NEXT:        Name:            bar_import
; NORMAL-NEXT:      - Index:           3
; NORMAL-NEXT:        Name:            hello
; NORMAL-NEXT:      - Index:           4
; NORMAL-NEXT:        Name:            my_func
; NORMAL-NEXT:      - Index:           5
; NORMAL-NEXT:        Name:            func_comdat
; NORMAL-NEXT:...
