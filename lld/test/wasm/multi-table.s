# RUN: not llvm-mc -filetype=obj -triple=wasm32-unknown-unknown -o %t.a1.o %s 2>&1 | FileCheck %s --check-prefix=MVP
# RUN: llvm-mc -filetype=obj -triple=wasm32-unknown-unknown -mattr=+reference-types -o %t.a1.rt.o %s
# RUN: llvm-mc -filetype=obj -triple=wasm32-unknown-unknown %p/Inputs/call-indirect.s -o %t.a2.o
# RUN: llvm-mc -filetype=obj -triple=wasm32-unknown-unknown -mattr=+reference-types %p/Inputs/call-indirect.s -o %t.a2.rt.o
# RUN: not wasm-ld --allow-undefined --export-dynamic --no-entry -o %t.wasm %t.a1.rt.o %t.a2.o 2>&1 | FileCheck %s --check-prefix=RT-MVP
# RUN: wasm-ld --allow-undefined --export-dynamic --no-entry -o- %t.a1.rt.o %t.a2.rt.o | obj2yaml | FileCheck %s

  .globl  table_a
  .tabletype table_a, funcref

  .globl  table_b
table_b:
  .tabletype table_b, funcref

  .globl  table_c
table_c:
  .tabletype table_c, externref

  .globl  call_indirect_explicit_tables
call_indirect_explicit_tables:
  .functype call_indirect_explicit_tables () -> ()
  i32.const 0
  call_indirect table_a, () -> ()
# MVP: error: Expected (, instead got: table_a
  i32.const 0
  call_indirect table_b, () -> ()
  end_function

# RT-MVP: wasm-ld: error: object file not built with 'reference-types' feature conflicts with import of table table_a by file

# CHECK:      --- !WASM
# CHECK-NEXT: FileHeader:
# CHECK-NEXT:   Version:         0x1
# CHECK-NEXT: Sections:
# CHECK-NEXT:   - Type:            TYPE
# CHECK-NEXT:     Signatures:
# CHECK-NEXT:       - Index:           0
# CHECK-NEXT:         ParamTypes:      []
# CHECK-NEXT:         ReturnTypes:     []
# CHECK-NEXT:       - Index:           1
# CHECK-NEXT:         ParamTypes:      []
# CHECK-NEXT:         ReturnTypes:
# CHECK-NEXT:           - I64
# CHECK-NEXT:       - Index:           2
# CHECK-NEXT:         ParamTypes:      []
# CHECK-NEXT:         ReturnTypes:
# CHECK-NEXT:           - I32
# CHECK-NEXT:   - Type:            IMPORT
# CHECK-NEXT:     Imports:
# CHECK-NEXT:       - Module:          env
# CHECK-NEXT:         Field:           table_a
# CHECK-NEXT:         Kind:            TABLE
# CHECK-NEXT:         Table:
# CHECK-NEXT:           Index:           0
# CHECK-NEXT:           ElemType:        FUNCREF
# CHECK-NEXT:           Limits:
# CHECK-NEXT:             Minimum:         0x0
# CHECK-NEXT:       - Module:          env
# CHECK-NEXT:         Field:           foo
# CHECK-NEXT:         Kind:            FUNCTION
# CHECK-NEXT:         SigIndex:        2
# CHECK-NEXT:   - Type:            FUNCTION
# CHECK-NEXT:     FunctionTypes:   [ 0, 1, 0 ]
# CHECK-NEXT:   - Type:            TABLE
# CHECK-NEXT:     Tables:
# CHECK-NEXT:       - Index:           1
# CHECK-NEXT:         ElemType:        FUNCREF
# CHECK-NEXT:         Limits:
# CHECK-NEXT:           Minimum:         0x0
# CHECK-NEXT:       - Index:           2
# CHECK-NEXT:         ElemType:        EXTERNREF
# CHECK-NEXT:         Limits:
# CHECK-NEXT:           Minimum:         0x0
# CHECK-NEXT:       - Index:           3
# CHECK-NEXT:         ElemType:        FUNCREF
# CHECK-NEXT:         Limits:
# CHECK-NEXT:           Flags:           [ HAS_MAX ]
# CHECK-NEXT:           Minimum:         0x3
# CHECK-NEXT:           Maximum:         0x3
# CHECK-NEXT:   - Type:            MEMORY
# CHECK-NEXT:     Memories:
# CHECK-NEXT:       - Minimum:         0x2
# CHECK-NEXT:   - Type:            GLOBAL
# CHECK-NEXT:     Globals:
# CHECK-NEXT:       - Index:           0
# CHECK-NEXT:         Type:            I32
# CHECK-NEXT:         Mutable:         true
# CHECK-NEXT:         InitExpr:
# CHECK-NEXT:           Opcode:          I32_CONST
# CHECK-NEXT:           Value:           66576
# CHECK-NEXT:   - Type:            EXPORT
# CHECK-NEXT:     Exports:
# CHECK-NEXT:       - Name:            memory
# CHECK-NEXT:         Kind:            MEMORY
# CHECK-NEXT:         Index:           0
# CHECK-NEXT:       - Name:            table_b
# CHECK-NEXT:         Kind:            TABLE
# CHECK-NEXT:         Index:           1
# CHECK-NEXT:       - Name:            table_c
# CHECK-NEXT:         Kind:            TABLE
# CHECK-NEXT:         Index:           2
# CHECK-NEXT:       - Name:            call_indirect_explicit_tables
# CHECK-NEXT:         Kind:            FUNCTION
# CHECK-NEXT:         Index:           1
# CHECK-NEXT:       - Name:            bar
# CHECK-NEXT:         Kind:            FUNCTION
# CHECK-NEXT:         Index:           2
# CHECK-NEXT:       - Name:            call_bar_indirect
# CHECK-NEXT:         Kind:            FUNCTION
# CHECK-NEXT:         Index:           3
# CHECK-NEXT:   - Type:            ELEM
# CHECK-NEXT:     Segments:
# CHECK-NEXT:       - Flags:           2
# CHECK-NEXT:         TableNumber:     3
# CHECK-NEXT:         ElemKind:        FUNCREF
# CHECK-NEXT:         Offset:
# CHECK-NEXT:           Opcode:          I32_CONST
# CHECK-NEXT:           Value:           1
# CHECK-NEXT:         Functions:       [ 2, 0 ]
# CHECK-NEXT:   - Type:            CODE
# CHECK-NEXT:     Functions:
# CHECK-NEXT:       - Index:           1
# CHECK-NEXT:         Locals:          []
# CHECK-NEXT:         Body:            41001180808080008080808000410011808080800081808080000B
# CHECK-NEXT:       - Index:           2
# CHECK-NEXT:         Locals:          []
# CHECK-NEXT:         Body:            42010B
# CHECK-NEXT:       - Index:           3
# CHECK-NEXT:         Locals:          []
# CHECK-NEXT:         Body:            41002802808880800011818080800083808080001A41002802848880800011828080800083808080001A0B
# CHECK-NEXT:   - Type:            DATA
# CHECK-NEXT:     Segments:
# CHECK-NEXT:       - SectionOffset:   7
# CHECK-NEXT:         InitFlags:       0
# CHECK-NEXT:         Offset:
# CHECK-NEXT:           Opcode:          I32_CONST
# CHECK-NEXT:           Value:           1024
# CHECK-NEXT:         Content:         '0100000002000000'
# CHECK-NEXT:   - Type:            CUSTOM
# CHECK-NEXT:     Name:            name
# CHECK-NEXT:     FunctionNames:
# CHECK-NEXT:       - Index:           0
# CHECK-NEXT:         Name:            foo
# CHECK-NEXT:       - Index:           1
# CHECK-NEXT:         Name:            call_indirect_explicit_tables
# CHECK-NEXT:       - Index:           2
# CHECK-NEXT:         Name:            bar
# CHECK-NEXT:       - Index:           3
# CHECK-NEXT:         Name:            call_bar_indirect
# CHECK-NEXT:     GlobalNames:
# CHECK-NEXT:       - Index:           0
# CHECK-NEXT:         Name:            __stack_pointer
