# RUN: llvm-mc -filetype=obj -triple=wasm32-unknown-unknown -o %t.o %s
# RUN: llvm-mc -filetype=obj -triple=wasm32-unknown-unknown %S/Inputs/weak-alias.s -o %t2.o
# RUN: wasm-ld --export-dynamic %t.o %t2.o -o %t.wasm
# RUN: obj2yaml %t.wasm | FileCheck %s

# Test that the strongly defined alias_fn from this file is used both here
# and in call_alias.

.globl alias_fn
alias_fn:
  .functype alias_fn () -> (i32)
  i32.const 1
  end_function

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
# CHECK-NEXT:         ParamTypes:      []
# CHECK-NEXT:         ReturnTypes:
# CHECK-NEXT:           - I32
# CHECK-NEXT:       - Index:           1
# CHECK-NEXT:         ParamTypes:      []
# CHECK-NEXT:         ReturnTypes:     []
# CHECK-NEXT:   - Type:            FUNCTION
# CHECK-NEXT:     FunctionTypes:   [ 0, 1, 0, 0, 0, 0, 0 ]
# CHECK-NEXT:   - Type:            TABLE
# CHECK-NEXT:     Tables:
# CHECK-NEXT:       - Index:           0
# CHECK-NEXT:         ElemType:        FUNCREF
# CHECK-NEXT:         Limits:
# CHECK-NEXT:           Flags:           [ HAS_MAX ]
# CHECK-NEXT:           Minimum:         0x3
# CHECK-NEXT:           Maximum:         0x3
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
# CHECK-NEXT:       - Name:            alias_fn
# CHECK-NEXT:         Kind:            FUNCTION
# CHECK-NEXT:         Index:           0
# CHECK-NEXT:       - Name:            _start
# CHECK-NEXT:         Kind:            FUNCTION
# CHECK-NEXT:         Index:           1
# CHECK-NEXT:       - Name:            direct_fn
# CHECK-NEXT:         Kind:            FUNCTION
# CHECK-NEXT:         Index:           2
# CHECK-NEXT:       - Name:            call_direct
# CHECK-NEXT:         Kind:            FUNCTION
# CHECK-NEXT:         Index:           3
# CHECK-NEXT:       - Name:            call_alias
# CHECK-NEXT:         Kind:            FUNCTION
# CHECK-NEXT:         Index:           4
# CHECK-NEXT:       - Name:            call_alias_ptr
# CHECK-NEXT:         Kind:            FUNCTION
# CHECK-NEXT:         Index:           5
# CHECK-NEXT:       - Name:            call_direct_ptr
# CHECK-NEXT:         Kind:            FUNCTION
# CHECK-NEXT:         Index:           6
# CHECK-NEXT:   - Type:            ELEM
# CHECK-NEXT:     Segments:
# CHECK-NEXT:       - Offset:
# CHECK-NEXT:           Opcode:          I32_CONST
# CHECK-NEXT:           Value:           1
# CHECK-NEXT:         Functions:       [ 0, 2 ]
# CHECK-NEXT:   - Type:            CODE
# CHECK-NEXT:     Functions:
# CHECK-NEXT:       - Index:           0
# CHECK-NEXT:         Locals:
# CHECK-NEXT:         Body:            41010B
# CHECK-NEXT:       - Index:           1
# CHECK-NEXT:         Locals:
# CHECK-NEXT:         Body:            1080808080001A0B
# CHECK-NEXT:       - Index:           2
# CHECK-NEXT:         Locals:
# CHECK-NEXT:         Body:            41000B
# CHECK-NEXT:       - Index:           3
# CHECK-NEXT:         Locals:
# CHECK-NEXT:         Body:            1082808080000B
# CHECK-NEXT:       - Index:           4
# CHECK-NEXT:         Locals:
# CHECK-NEXT:         Body:            1080808080000B
# CHECK-NEXT:       - Index:           5
# CHECK-NEXT:         Locals:
# CHECK-NEXT:         Body:            41818080800011808080800080808080000B
# CHECK-NEXT:       - Index:           6
# CHECK-NEXT:         Locals:
# CHECK-NEXT:         Body:            41828080800011808080800080808080000B
# CHECK-NEXT:   - Type:            CUSTOM
# CHECK-NEXT:     Name:            name
# CHECK-NEXT:     FunctionNames:
# CHECK-NEXT:       - Index:           0
# CHECK-NEXT:         Name:            alias_fn
# CHECK-NEXT:       - Index:           1
# CHECK-NEXT:         Name:            _start
# CHECK-NEXT:       - Index:           2
# CHECK-NEXT:         Name:            direct_fn
# CHECK-NEXT:       - Index:           3
# CHECK-NEXT:         Name:            call_direct
# CHECK-NEXT:       - Index:           4
# CHECK-NEXT:         Name:            call_alias
# CHECK-NEXT:       - Index:           5
# CHECK-NEXT:         Name:            call_alias_ptr
# CHECK-NEXT:       - Index:           6
# CHECK-NEXT:         Name:            call_direct_ptr
# CHECK-NEXT:     GlobalNames:
# CHECK-NEXT:       - Index:           0
# CHECK-NEXT:         Name:            __stack_pointer
# CHECK-NEXT: ...
