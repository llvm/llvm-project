# RUN: llvm-mc -filetype=obj -triple=wasm32-unknown-unknown -o %t.o %s
# RUN: llvm-mc -filetype=obj -triple=wasm32-unknown-unknown -o %t2.o %S/Inputs/weak-alias.s
# RUN: wasm-ld --export-dynamic %t.o %t2.o -o %t.wasm
# RUN: obj2yaml %t.wasm | FileCheck %s

# Test that weak aliases (alias_fn is a weak alias of direct_fn) are linked correctly

.functype alias_fn () -> (i32)

.globl _start
_start:
  .functype _start () -> ()
  call alias_fn
  drop
  end_function

# CHECK:      --- !WASM
# CHECK-NEXT: FileHeader:
# CHECK-NEXT:   Version:         0x1
# CHECK-NEXT: Sections:
# CHECK-NEXT:   - Type:            TYPE
# CHECK-NEXT:     Signatures:
# CHECK-NEXT:       - Index:           0
# CHECK-NEXT:         ParamTypes:
# CHECK-NEXT:         ReturnTypes:
# CHECK-NEXT:           - I32
# CHECK-NEXT:       - Index:           1
# CHECK-NEXT:         ParamTypes:      []
# CHECK-NEXT:         ReturnTypes:     []
# CHECK-NEXT:   - Type:            FUNCTION
# CHECK-NEXT:     FunctionTypes:   [ 1, 0, 0, 0, 0, 0 ]
# CHECK-NEXT:   - Type:            TABLE
# CHECK-NEXT:     Tables:
# CHECK-NEXT:       - Index:           0
# CHECK-NEXT:         ElemType:        FUNCREF
# CHECK-NEXT:         Limits:
# CHECK-NEXT:           Flags:           [ HAS_MAX ]
# CHECK-NEXT:           Minimum:         0x2
# CHECK-NEXT:           Maximum:         0x2
# CHECK-NEXT:   - Type:            MEMORY
# CHECK-NEXT:     Memories:
# CHECK-NEXT:       - Minimum:         0x1
# CHECK-NEXT:   - Type:            GLOBAL
# CHECK-NEXT:     Globals:
# CHECK-NEXT:       - Index:           0
# CHECK-NEXT:         Type:            I32
# CHECK-NEXT:         Mutable:         true
# CHECK-NEXT:         InitExpr:
# CHECK-NEXT:           Opcode:          I32_CONST
# CHECK-NEXT:           Value:           65536
# CHECK-NEXT:   - Type:            EXPORT
# CHECK-NEXT:     Exports:
# CHECK-NEXT:       - Name:            memory
# CHECK-NEXT:         Kind:            MEMORY
# CHECK-NEXT:         Index:           0
# CHECK-NEXT:       - Name:            _start
# CHECK-NEXT:         Kind:            FUNCTION
# CHECK-NEXT:         Index:           0
# CHECK-NEXT:       - Name:            alias_fn
# CHECK-NEXT:         Kind:            FUNCTION
# CHECK-NEXT:         Index:           1
# CHECK-NEXT:       - Name:            direct_fn
# CHECK-NEXT:         Kind:            FUNCTION
# CHECK-NEXT:         Index:           1
# CHECK-NEXT:       - Name:            call_direct
# CHECK-NEXT:         Kind:            FUNCTION
# CHECK-NEXT:         Index:           2
# CHECK-NEXT:       - Name:            call_alias
# CHECK-NEXT:         Kind:            FUNCTION
# CHECK-NEXT:         Index:           3
# CHECK-NEXT:       - Name:            call_alias_ptr
# CHECK-NEXT:         Kind:            FUNCTION
# CHECK-NEXT:         Index:           4
# CHECK-NEXT:       - Name:            call_direct_ptr
# CHECK-NEXT:         Kind:            FUNCTION
# CHECK-NEXT:         Index:           5
# CHECK-NEXT:   - Type:            ELEM
# CHECK-NEXT:     Segments:
# CHECK-NEXT:       - Offset:
# CHECK-NEXT:           Opcode:          I32_CONST
# CHECK-NEXT:           Value:           1
# CHECK-NEXT:         Functions:       [ 1 ]
# CHECK-NEXT:   - Type:            CODE
# CHECK-NEXT:     Functions:
# CHECK-NEXT:       - Index:           0
# CHECK-NEXT:         Locals:
# CHECK-NEXT:         Body:            1081808080001A0B
# CHECK-NEXT:       - Index:           1
# CHECK-NEXT:         Locals:
# CHECK-NEXT:         Body:            41000B
# CHECK-NEXT:       - Index:           2
# CHECK-NEXT:         Locals:
# CHECK-NEXT:         Body:            1081808080000B
# CHECK-NEXT:       - Index:           3
# CHECK-NEXT:         Locals:
# CHECK-NEXT:         Body:            1081808080000B
# CHECK-NEXT:       - Index:           4
# CHECK-NEXT:         Locals:
# CHECK-NEXT:         Body:            41818080800011808080800080808080000B
# CHECK-NEXT:       - Index:           5
# CHECK-NEXT:         Locals:
# CHECK-NEXT:         Body:            41818080800011808080800080808080000B
# CHECK-NEXT:   - Type:            CUSTOM
# CHECK-NEXT:     HeaderSecSizeEncodingLen: 2
# CHECK-NEXT:     Name:            name
# CHECK-NEXT:     FunctionNames:
# CHECK-NEXT:       - Index:           0
# CHECK-NEXT:         Name:            _start
# CHECK-NEXT:       - Index:           1
# CHECK-NEXT:         Name:            direct_fn
# CHECK-NEXT:       - Index:           2
# CHECK-NEXT:         Name:            call_direct
# CHECK-NEXT:       - Index:           3
# CHECK-NEXT:         Name:            call_alias
# CHECK-NEXT:       - Index:           4
# CHECK-NEXT:         Name:            call_alias_ptr
# CHECK-NEXT:       - Index:           5
# CHECK-NEXT:         Name:            call_direct_ptr
# CHECK-NEXT:     GlobalNames:
# CHECK-NEXT:       - Index:           0
# CHECK-NEXT:         Name:            __stack_pointer
# CHECK-NEXT: ...

# RUN: wasm-ld --relocatable %t.o %t2.o -o %t.reloc.o
# RUN: obj2yaml %t.reloc.o | FileCheck %s -check-prefix=RELOC

# RELOC:      --- !WASM
# RELOC-NEXT: FileHeader:
# RELOC-NEXT:   Version:         0x1
# RELOC-NEXT: Sections:
# RELOC-NEXT:   - Type:            TYPE
# RELOC-NEXT:     Signatures:
# RELOC-NEXT:       - Index:           0
# RELOC-NEXT:         ParamTypes:      []
# RELOC-NEXT:         ReturnTypes:
# RELOC-NEXT:           - I32
# RELOC-NEXT:       - Index:           1
# RELOC-NEXT:         ParamTypes:      []
# RELOC-NEXT:         ReturnTypes:     []
# RELOC-NEXT:   - Type:            IMPORT
# RELOC-NEXT:     Imports:
# RELOC-NEXT:       - Module:          env
# RELOC-NEXT:         Field:           __indirect_function_table
# RELOC-NEXT:         Kind:            TABLE
# RELOC-NEXT:         Table:
# RELOC-NEXT:           Index:           0
# RELOC-NEXT:           ElemType:        FUNCREF
# RELOC-NEXT:           Limits:
# RELOC-NEXT:             Minimum:         0x2
# RELOC-NEXT:   - Type:            FUNCTION
# RELOC-NEXT:     FunctionTypes:   [ 1, 0, 0, 0, 0, 0 ]
# RELOC-NEXT:   - Type:            MEMORY
# RELOC-NEXT:     Memories:
# RELOC-NEXT:       - Minimum:         0x0
# RELOC-NEXT:   - Type:            ELEM
# RELOC-NEXT:     Segments:
# RELOC-NEXT:       - Offset:
# RELOC-NEXT:           Opcode:          I32_CONST
# RELOC-NEXT:           Value:           1
# RELOC-NEXT:         Functions:       [ 1 ]
# RELOC-NEXT:   - Type:            CODE
# RELOC-NEXT:     Relocations:
# RELOC-NEXT:       - Type:            R_WASM_FUNCTION_INDEX_LEB
# RELOC-NEXT:         Index:           1
# RELOC-NEXT:         Offset:          0x4
# RELOC-NEXT:       - Type:            R_WASM_FUNCTION_INDEX_LEB
# RELOC-NEXT:         Index:           2
# RELOC-NEXT:         Offset:          0x13
# RELOC-NEXT:       - Type:            R_WASM_FUNCTION_INDEX_LEB
# RELOC-NEXT:         Index:           1
# RELOC-NEXT:         Offset:          0x1C
# RELOC-NEXT:       - Type:            R_WASM_TABLE_INDEX_SLEB
# RELOC-NEXT:         Index:           1
# RELOC-NEXT:         Offset:          0x25
# RELOC-NEXT:       - Type:            R_WASM_TYPE_INDEX_LEB
# RELOC-NEXT:         Index:           0
# RELOC-NEXT:         Offset:          0x2B
# RELOC-NEXT:       - Type:            R_WASM_TABLE_NUMBER_LEB
# RELOC-NEXT:         Index:           6
# RELOC-NEXT:         Offset:          0x30
# RELOC-NEXT:       - Type:            R_WASM_TABLE_INDEX_SLEB
# RELOC-NEXT:         Index:           2
# RELOC-NEXT:         Offset:          0x39
# RELOC-NEXT:       - Type:            R_WASM_TYPE_INDEX_LEB
# RELOC-NEXT:         Index:           0
# RELOC-NEXT:         Offset:          0x3F
# RELOC-NEXT:       - Type:            R_WASM_TABLE_NUMBER_LEB
# RELOC-NEXT:         Index:           6
# RELOC-NEXT:         Offset:          0x44
# RELOC-NEXT:     Functions:
# RELOC-NEXT:       - Index:           0
# RELOC-NEXT:         Locals:
# RELOC-NEXT:         Body:            1081808080001A0B
# RELOC-NEXT:       - Index:           1
# RELOC-NEXT:         Locals:
# RELOC-NEXT:         Body:            41000B
# RELOC-NEXT:       - Index:           2
# RELOC-NEXT:         Locals:
# RELOC-NEXT:         Body:            1081808080000B
# RELOC-NEXT:       - Index:           3
# RELOC-NEXT:         Locals:
# RELOC-NEXT:         Body:            1081808080000B
# RELOC-NEXT:       - Index:           4
# RELOC-NEXT:         Locals:
# RELOC-NEXT:         Body:            41818080800011808080800080808080000B
# RELOC-NEXT:       - Index:           5
# RELOC-NEXT:         Locals:
# RELOC-NEXT:         Body:            41818080800011808080800080808080000B
# RELOC-NEXT:   - Type:            CUSTOM
# RELOC-NEXT:     Name:            linking
# RELOC-NEXT:     Version:         2
# RELOC-NEXT:     SymbolTable:
# RELOC-NEXT:       - Index:           0
# RELOC-NEXT:         Kind:            FUNCTION
# RELOC-NEXT:         Name:            _start
# RELOC-NEXT:         Flags:           [  ]
# RELOC-NEXT:         Function:        0
# RELOC-NEXT:       - Index:           1
# RELOC-NEXT:         Kind:            FUNCTION
# RELOC-NEXT:         Name:            alias_fn
# RELOC-NEXT:         Flags:           [ BINDING_WEAK ]
# RELOC-NEXT:         Function:        1
# RELOC-NEXT:       - Index:           2
# RELOC-NEXT:         Kind:            FUNCTION
# RELOC-NEXT:         Name:            direct_fn
# RELOC-NEXT:         Flags:           [  ]
# RELOC-NEXT:         Function:        1
# RELOC-NEXT:       - Index:           3
# RELOC-NEXT:         Kind:            FUNCTION
# RELOC-NEXT:         Name:            call_direct
# RELOC-NEXT:         Flags:           [  ]
# RELOC-NEXT:         Function:        2
# RELOC-NEXT:       - Index:           4
# RELOC-NEXT:         Kind:            FUNCTION
# RELOC-NEXT:         Name:            call_alias
# RELOC-NEXT:         Flags:           [  ]
# RELOC-NEXT:         Function:        3
# RELOC-NEXT:       - Index:           5
# RELOC-NEXT:         Kind:            FUNCTION
# RELOC-NEXT:         Name:            call_alias_ptr
# RELOC-NEXT:         Flags:           [  ]
# RELOC-NEXT:         Function:        4
# RELOC-NEXT:       - Index:           6
# RELOC-NEXT:         Kind:            TABLE
# RELOC-NEXT:         Name:            __indirect_function_table
# RELOC-NEXT:         Flags:           [ UNDEFINED, NO_STRIP ]
# RELOC-NEXT:         Table:           0
# RELOC-NEXT:       - Index:           7
# RELOC-NEXT:         Kind:            FUNCTION
# RELOC-NEXT:         Name:            call_direct_ptr
# RELOC-NEXT:         Flags:           [  ]
# RELOC-NEXT:         Function:        5
# RELOC-NEXT:   - Type:            CUSTOM
# RELOC-NEXT:     Name:            name
# RELOC-NEXT:     FunctionNames:
# RELOC-NEXT:       - Index:           0
# RELOC-NEXT:         Name:            _start
# RELOC-NEXT:       - Index:           1
# RELOC-NEXT:         Name:            direct_fn
# RELOC-NEXT:       - Index:           2
# RELOC-NEXT:         Name:            call_direct
# RELOC-NEXT:       - Index:           3
# RELOC-NEXT:         Name:            call_alias
# RELOC-NEXT:       - Index:           4
# RELOC-NEXT:         Name:            call_alias_ptr
# RELOC-NEXT:       - Index:           5
# RELOC-NEXT:         Name:            call_direct_ptr
# RELOC-NEXT: ...
