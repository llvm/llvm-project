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

@func_addr1 = hidden global ptr @my_func, align 4
@func_addr2 = hidden global ptr @foo_import, align 4
@func_addr3 = hidden global ptr @bar_import, align 4
@data_addr1 = hidden global ptr @data_import, align 8

$func_comdat = comdat any
@data_comdat = weak_odr constant [3 x i8] c"abc", comdat($func_comdat)
define linkonce_odr i32 @func_comdat() comdat {
entry:
  ret i32 ptrtoint (ptr @data_comdat to i32)
}

; Test that __attribute__(used) (i.e NO_STRIP) is preserved in the relocated symbol table
@llvm.used = appending global [1 x ptr] [ptr @my_func], section "llvm.metadata"

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
; CHECK-NEXT:             Minimum:         0x4
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
; CHECK-NEXT:      - Minimum:         0x1
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
; CHECK-NEXT:       - Index:           6
; CHECK-NEXT:         Locals:          []
; CHECK-NEXT:         Body:            0B
; CHECK-NEXT:  - Type:            DATA
; CHECK-NEXT:    Relocations:
; CHECK-NEXT:      - Type:            R_WASM_TABLE_INDEX_I32
; CHECK-NEXT:        Index:           3
; CHECK-NEXT:        Offset:          0x1A
; CHECK-NEXT:      - Type:            R_WASM_TABLE_INDEX_I32
; CHECK-NEXT:        Index:           4
; CHECK-NEXT:        Offset:          0x23
; CHECK-NEXT:      - Type:            R_WASM_TABLE_INDEX_I32
; CHECK-NEXT:        Index:           5
; CHECK-NEXT:        Offset:          0x2C
; CHECK-NEXT:      - Type:            R_WASM_MEMORY_ADDR_I32
; CHECK-NEXT:        Index:           13
; CHECK-NEXT:        Offset:          0x35
; CHECK-NEXT:    Segments:
; CHECK-NEXT:      - SectionOffset:   6
; CHECK-NEXT:        InitFlags:       0
; CHECK-NEXT:        Offset:
; CHECK-NEXT:          Opcode:          I32_CONST
; CHECK-NEXT:          Value:           0
; CHECK-NEXT:        Content:         68656C6C6F0A00
; CHECK-NEXT:      - SectionOffset:   18
; CHECK-NEXT:        InitFlags:       0
; CHECK-NEXT:        Offset:
; CHECK-NEXT:          Opcode:          I32_CONST
; CHECK-NEXT:          Value:           7
; CHECK-NEXT:        Content:         '616263'
; CHECK-NEXT:      - SectionOffset:   26
; CHECK-NEXT:        InitFlags:       0
; CHECK-NEXT:        Offset:
; CHECK-NEXT:          Opcode:          I32_CONST
; CHECK-NEXT:          Value:           12
; CHECK-NEXT:        Content:         '01000000'
; CHECK-NEXT:      - SectionOffset:   35
; CHECK-NEXT:        InitFlags:       0
; CHECK-NEXT:        Offset:
; CHECK-NEXT:          Opcode:          I32_CONST
; CHECK-NEXT:          Value:           16
; CHECK-NEXT:        Content:         '02000000'
; CHECK-NEXT:      - SectionOffset:   44
; CHECK-NEXT:        InitFlags:       0
; CHECK-NEXT:        Offset:
; CHECK-NEXT:          Opcode:          I32_CONST
; CHECK-NEXT:          Value:           20
; CHECK-NEXT:        Content:         '03000000'
; CHECK-NEXT:      - SectionOffset:   53
; CHECK-NEXT:        InitFlags:       0
; CHECK-NEXT:        Offset:
; CHECK-NEXT:          Opcode:          I32_CONST
; CHECK-NEXT:          Value:           24
; CHECK-NEXT:        Content:         '00000000'
; CHECK-NEXT:  - Type:            CUSTOM
; CHECK-NEXT:    Name:            linking
; CHECK-NEXT:    Version:         2
; CHECK-NEXT:    SymbolTable:
; CHECK-NEXT:      - Index:           0
; CHECK-NEXT:        Kind:            FUNCTION
; CHECK-NEXT:        Name:            hello
; CHECK-NEXT:        Flags:           [  ]
; CHECK-NEXT:        Function:        3
; CHECK-NEXT:      - Index:           1
; CHECK-NEXT:        Kind:            DATA
; CHECK-NEXT:        Name:            hello_str
; CHECK-NEXT:        Flags:           [  ]
; CHECK-NEXT:        Segment:         0
; CHECK-NEXT:        Size:            7
; CHECK-NEXT:      - Index:           2
; CHECK-NEXT:        Kind:            FUNCTION
; CHECK-NEXT:        Name:            puts
; CHECK-NEXT:        Flags:           [ UNDEFINED ]
; CHECK-NEXT:        Function:        0
; CHECK-NEXT:      - Index:           3
; CHECK-NEXT:        Kind:            FUNCTION
; CHECK-NEXT:        Name:            my_func
; CHECK-NEXT:        Flags:           [ VISIBILITY_HIDDEN, NO_STRIP ]
; CHECK-NEXT:        Function:        4
; CHECK-NEXT:      - Index:           4
; CHECK-NEXT:        Kind:            FUNCTION
; CHECK-NEXT:        Name:            foo_import
; CHECK-NEXT:        Flags:           [ UNDEFINED ]
; CHECK-NEXT:        Function:        1
; CHECK-NEXT:      - Index:           5
; CHECK-NEXT:        Kind:            FUNCTION
; CHECK-NEXT:        Name:            bar_import
; CHECK-NEXT:        Flags:           [ BINDING_WEAK, UNDEFINED ]
; CHECK-NEXT:        Function:        2
; CHECK-NEXT:      - Index:           6
; CHECK-NEXT:        Kind:            FUNCTION
; CHECK-NEXT:        Name:            func_comdat
; CHECK-NEXT:        Flags:           [ BINDING_WEAK ]
; CHECK-NEXT:        Function:        5
; CHECK-NEXT:      - Index:           7
; CHECK-NEXT:        Kind:            DATA
; CHECK-NEXT:        Name:            data_comdat
; CHECK-NEXT:        Flags:           [ BINDING_WEAK ]
; CHECK-NEXT:        Segment:         1
; CHECK-NEXT:        Size:            3
; CHECK-NEXT:      - Index:           8
; CHECK-NEXT:        Kind:            FUNCTION
; CHECK-NEXT:        Name:            _start
; CHECK-NEXT:        Flags:           [ ]
; CHECK-NEXT:        Function:        6
; CHECK-NEXT:      - Index:           9
; CHECK-NEXT:        Kind:            DATA
; CHECK-NEXT:        Name:            func_addr1
; CHECK-NEXT:        Flags:           [ VISIBILITY_HIDDEN ]
; CHECK-NEXT:        Segment:         2
; CHECK-NEXT:        Size:            4
; CHECK-NEXT:      - Index:           10
; CHECK-NEXT:        Kind:            DATA
; CHECK-NEXT:        Name:            func_addr2
; CHECK-NEXT:        Flags:           [ VISIBILITY_HIDDEN ]
; CHECK-NEXT:        Segment:         3
; CHECK-NEXT:        Size:            4
; CHECK-NEXT:      - Index:           11
; CHECK-NEXT:        Kind:            DATA
; CHECK-NEXT:        Name:            func_addr3
; CHECK-NEXT:        Flags:           [ VISIBILITY_HIDDEN ]
; CHECK-NEXT:        Segment:         4
; CHECK-NEXT:        Size:            4
; CHECK-NEXT:      - Index:           12
; CHECK-NEXT:        Kind:            DATA
; CHECK-NEXT:        Name:            data_addr1
; CHECK-NEXT:        Flags:           [ VISIBILITY_HIDDEN ]
; CHECK-NEXT:        Segment:         5
; CHECK-NEXT:        Size:            4
; CHECK-NEXT:      - Index:           13
; CHECK-NEXT:        Kind:            DATA
; CHECK-NEXT:        Name:            data_import
; CHECK-NEXT:        Flags:           [ UNDEFINED ]
; CHECK-NEXT:      - Index:           14
; CHECK-NEXT:        Kind:            TABLE
; CHECK-NEXT:        Name:            __indirect_function_table
; CHECK-NEXT:        Flags:           [ UNDEFINED, NO_STRIP ]
; CHECK-NEXT:        Table:           0
; CHECK-NEXT:    SegmentInfo:
; CHECK-NEXT:      - Index:           0
; CHECK-NEXT:        Name:            .rodata.hello_str
; CHECK-NEXT:        Alignment:       0
; CHECK-NEXT:        Flags:           [  ]
; CHECK-NEXT:      - Index:           1
; CHECK-NEXT:        Name:            .rodata.data_comdat
; CHECK-NEXT:        Alignment:       0
; CHECK-NEXT:        Flags:           [  ]
; CHECK-NEXT:      - Index:           2
; CHECK-NEXT:        Name:            .data.func_addr1
; CHECK-NEXT:        Alignment:       2
; CHECK-NEXT:        Flags:           [  ]
; CHECK-NEXT:      - Index:           3
; CHECK-NEXT:        Name:            .data.func_addr2
; CHECK-NEXT:        Alignment:       2
; CHECK-NEXT:        Flags:           [  ]
; CHECK-NEXT:      - Index:           4
; CHECK-NEXT:        Name:            .data.func_addr3
; CHECK-NEXT:        Alignment:       2
; CHECK-NEXT:        Flags:           [  ]
; CHECK-NEXT:      - Index:           5
; CHECK-NEXT:        Name:            .data.data_addr1
; CHECK-NEXT:        Alignment:       3
; CHECK-NEXT:        Flags:           [  ]
; CHECK-NEXT:    Comdats:
; CHECK-NEXT:      - Name:            func_comdat
; CHECK-NEXT:        Entries:
; CHECK-NEXT:          - Kind:            FUNCTION
; CHECK-NEXT:            Index:           5
; CHECK-NEXT:          - Kind:            DATA
; CHECK-NEXT:            Index:           1
; CHECK-NEXT:  - Type:            CUSTOM
; CHECK-NEXT:    Name:            name
; CHECK-NEXT:    FunctionNames:
; CHECK-NEXT:      - Index:           0
; CHECK-NEXT:        Name:            puts
; CHECK-NEXT:      - Index:           1
; CHECK-NEXT:        Name:            foo_import
; CHECK-NEXT:      - Index:           2
; CHECK-NEXT:        Name:            bar_import
; CHECK-NEXT:      - Index:           3
; CHECK-NEXT:        Name:            hello
; CHECK-NEXT:      - Index:           4
; CHECK-NEXT:        Name:            my_func
; CHECK-NEXT:      - Index:           5
; CHECK-NEXT:        Name:            func_comdat
; CHECK-NEXT:      - Index:           6
; CHECK-NEXT:        Name:            _start
; CHECK-NEXT:    DataSegmentNames:
; CHECK-NEXT:      - Index:           0
; CHECK-NEXT:        Name:            .rodata.hello_str
; CHECK-NEXT:      - Index:           1
; CHECK-NEXT:        Name:            .rodata.data_comdat
; CHECK-NEXT:      - Index:           2
; CHECK-NEXT:        Name:            .data.func_addr1
; CHECK-NEXT:      - Index:           3
; CHECK-NEXT:        Name:            .data.func_addr2
; CHECK-NEXT:      - Index:           4
; CHECK-NEXT:        Name:            .data.func_addr3
; CHECK-NEXT:      - Index:           5
; CHECK-NEXT:        Name:            .data.data_addr1
; CHECK-NEXT:  - Type:            CUSTOM
