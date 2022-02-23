# RUN: llvm-mc -filetype=obj -triple=wasm32-unknown-unknown -o %t.o %s
# RUN: wasm-ld --relocatable -o %t.wasm %t.o
# RUN: obj2yaml %t.wasm | FileCheck %s

.globaltype __stack_pointer, i32

.globl  _start
_start:
  .functype _start () -> (i32)
  global.get __stack_pointer
  i32.const 16
  i32.sub
  drop
  i32.const 0
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
# CHECK-NEXT:   - Type:            IMPORT
# CHECK-NEXT:     Imports:
# CHECK-NEXT:       - Module:          env
# CHECK-NEXT:         Field:           __stack_pointer
# CHECK-NEXT:         Kind:            GLOBAL
# CHECK-NEXT:         GlobalType:      I32
# CHECK-NEXT:         GlobalMutable:   true
# CHECK-NEXT:   - Type:            FUNCTION
# CHECK-NEXT:     FunctionTypes:   [ 0 ]
# CHECK-NEXT:   - Type:            MEMORY
# CHECK-NEXT:     Memories:
# CHECK-NEXT:       - Minimum:         0x0
# CHECK-NEXT:   - Type:            CODE
# CHECK-NEXT:     Relocations:
# CHECK-NEXT:       - Type:            R_WASM_GLOBAL_INDEX_LEB
# CHECK-NEXT:         Index:           1
# CHECK-NEXT:         Offset:          0x4
# CHECK-NEXT:     Functions:
# CHECK-NEXT:       - Index:           0
# CHECK-NEXT:         Locals:
# CHECK-NEXT:         Body:            23808080800041106B1A41000B
# CHECK-NEXT:   - Type:            CUSTOM
# CHECK-NEXT:     Name:            linking
# CHECK-NEXT:     Version:         2
# CHECK-NEXT:     SymbolTable:
# CHECK-NEXT:       - Index:           0
# CHECK-NEXT:         Kind:            FUNCTION
# CHECK-NEXT:         Name:            _start
# CHECK-NEXT:         Flags:           [  ]
# CHECK-NEXT:         Function:        0
# CHECK-NEXT:       - Index:           1
# CHECK-NEXT:         Kind:            GLOBAL
# CHECK-NEXT:         Name:            __stack_pointer
# CHECK-NEXT:         Flags:           [ UNDEFINED ]
# CHECK-NEXT:         Global:          0
# CHECK-NEXT:   - Type:            CUSTOM
# CHECK-NEXT:     Name:            name
# CHECK-NEXT:     FunctionNames:
# CHECK-NEXT:       - Index:           0
# CHECK-NEXT:         Name:            _start
# CHECK-NEXT:     GlobalNames:
# CHECK-NEXT:       - Index:           0
# CHECK-NEXT:         Name:            __stack_pointer
# CHECK-NEXT: ...
