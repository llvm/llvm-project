# RUN: llvm-mc -filetype=obj -triple=wasm32-unknown-unknown %s -o %t.o

# RUN: wasm-ld -o %t.wasm %t.o
# RUN: obj2yaml %t.wasm | FileCheck %s  -check-prefix=CHECK-DEFAULT

# RUN: wasm-ld --function-pointer-alignment=2 -o %t.wasm %t.o
# RUN: obj2yaml %t.wasm | FileCheck %s  -check-prefix=CHECK-2

# RUN: wasm-ld --function-pointer-alignment=3 -o %t.wasm %t.o
# RUN: obj2yaml %t.wasm | FileCheck %s  -check-prefix=CHECK-3

.functype func1 () -> ()
.functype func2 () -> ()

.globl _start
_start:
  .functype _start () -> ()
  i32.const func1
  drop
  i32.const func2
  drop
  end_function

.globl func1
func1:
  .functype func1 () -> ()
  end_function

.globl func2
func2:
  .functype func2 () -> ()
  end_function


# CHECK-DEFAULT:      --- !WASM
# CHECK-DEFAULT-NEXT: FileHeader:
# CHECK-DEFAULT-NEXT:   Version:         0x1
# CHECK-DEFAULT-NEXT: Sections:
# CHECK-DEFAULT-NEXT:   - Type:            TYPE
# CHECK-DEFAULT-NEXT:     Signatures:
# CHECK-DEFAULT-NEXT:       - Index:           0
# CHECK-DEFAULT-NEXT:         ParamTypes:      []
# CHECK-DEFAULT-NEXT:         ReturnTypes:     []
# CHECK-DEFAULT-NEXT:   - Type:            FUNCTION
# CHECK-DEFAULT-NEXT:     FunctionTypes:   [ 0, 0, 0 ]
# CHECK-DEFAULT-NEXT:   - Type:            TABLE
# CHECK-DEFAULT-NEXT:     Tables:
# CHECK-DEFAULT-NEXT:       - Index:           0
# CHECK-DEFAULT-NEXT:         ElemType:        FUNCREF
# CHECK-DEFAULT-NEXT:         Limits:
# CHECK-DEFAULT-NEXT:           Flags:           [ HAS_MAX ]
# CHECK-DEFAULT-NEXT:           Minimum:         0x3
# CHECK-DEFAULT-NEXT:           Maximum:         0x3
# CHECK-DEFAULT-NEXT:   - Type:            MEMORY
# CHECK-DEFAULT-NEXT:     Memories:
# CHECK-DEFAULT-NEXT:       - Minimum:         0x2
# CHECK-DEFAULT-NEXT:   - Type:            GLOBAL
# CHECK-DEFAULT-NEXT:     Globals:
# CHECK-DEFAULT-NEXT:       - Index:           0
# CHECK-DEFAULT-NEXT:         Type:            I32
# CHECK-DEFAULT-NEXT:         Mutable:         true
# CHECK-DEFAULT-NEXT:         InitExpr:
# CHECK-DEFAULT-NEXT:           Opcode:          I32_CONST
# CHECK-DEFAULT-NEXT:           Value:           66560
# CHECK-DEFAULT-NEXT:   - Type:            EXPORT
# CHECK-DEFAULT-NEXT:     Exports:
# CHECK-DEFAULT-NEXT:       - Name:            memory
# CHECK-DEFAULT-NEXT:         Kind:            MEMORY
# CHECK-DEFAULT-NEXT:         Index:           0
# CHECK-DEFAULT-NEXT:       - Name:            _start
# CHECK-DEFAULT-NEXT:         Kind:            FUNCTION
# CHECK-DEFAULT-NEXT:         Index:           0
# CHECK-DEFAULT-NEXT:   - Type:            ELEM
# CHECK-DEFAULT-NEXT:     Segments:
# CHECK-DEFAULT-NEXT:       - Offset:
# CHECK-DEFAULT-NEXT:           Opcode:          I32_CONST
# CHECK-DEFAULT-NEXT:           Value:           1
# CHECK-DEFAULT-NEXT:         Functions:       [ 1, 2 ]
# CHECK-DEFAULT-NEXT:   - Type:            CODE
# CHECK-DEFAULT-NEXT:     Functions:
# CHECK-DEFAULT-NEXT:       - Index:           0
# CHECK-DEFAULT-NEXT:         Locals:          []
# CHECK-DEFAULT-NEXT:         Body:            4181808080001A4182808080001A0B
# CHECK-DEFAULT-NEXT:       - Index:           1
# CHECK-DEFAULT-NEXT:         Locals:          []
# CHECK-DEFAULT-NEXT:         Body:            0B
# CHECK-DEFAULT-NEXT:       - Index:           2
# CHECK-DEFAULT-NEXT:         Locals:          []
# CHECK-DEFAULT-NEXT:         Body:            0B
# CHECK-DEFAULT-NEXT:   - Type:            CUSTOM
# CHECK-DEFAULT-NEXT:     Name:            name
# CHECK-DEFAULT-NEXT:     FunctionNames:
# CHECK-DEFAULT-NEXT:       - Index:           0
# CHECK-DEFAULT-NEXT:         Name:            _start
# CHECK-DEFAULT-NEXT:       - Index:           1
# CHECK-DEFAULT-NEXT:         Name:            func1
# CHECK-DEFAULT-NEXT:       - Index:           2
# CHECK-DEFAULT-NEXT:         Name:            func2
# CHECK-DEFAULT-NEXT:     GlobalNames:
# CHECK-DEFAULT-NEXT:       - Index:           0
# CHECK-DEFAULT-NEXT:         Name:            __stack_pointer

# CHECK-2:      --- !WASM
# CHECK-2-NEXT: FileHeader:
# CHECK-2-NEXT:   Version:         0x1
# CHECK-2-NEXT: Sections:
# CHECK-2-NEXT:   - Type:            TYPE
# CHECK-2-NEXT:     Signatures:
# CHECK-2-NEXT:       - Index:           0
# CHECK-2-NEXT:         ParamTypes:      []
# CHECK-2-NEXT:         ReturnTypes:     []
# CHECK-2-NEXT:   - Type:            FUNCTION
# CHECK-2-NEXT:     FunctionTypes:   [ 0, 0, 0 ]
# CHECK-2-NEXT:   - Type:            TABLE
# CHECK-2-NEXT:     Tables:
# CHECK-2-NEXT:       - Index:           0
# CHECK-2-NEXT:         ElemType:        FUNCREF
# CHECK-2-NEXT:         Limits:
# CHECK-2-NEXT:           Flags:           [ HAS_MAX ]
# CHECK-2-NEXT:           Minimum:         0x4
# CHECK-2-NEXT:           Maximum:         0x4
# CHECK-2-NEXT:   - Type:            MEMORY
# CHECK-2-NEXT:     Memories:
# CHECK-2-NEXT:       - Minimum:         0x2
# CHECK-2-NEXT:   - Type:            GLOBAL
# CHECK-2-NEXT:     Globals:
# CHECK-2-NEXT:       - Index:           0
# CHECK-2-NEXT:         Type:            I32
# CHECK-2-NEXT:         Mutable:         true
# CHECK-2-NEXT:         InitExpr:
# CHECK-2-NEXT:           Opcode:          I32_CONST
# CHECK-2-NEXT:           Value:           66560
# CHECK-2-NEXT:   - Type:            EXPORT
# CHECK-2-NEXT:     Exports:
# CHECK-2-NEXT:       - Name:            memory
# CHECK-2-NEXT:         Kind:            MEMORY
# CHECK-2-NEXT:         Index:           0
# CHECK-2-NEXT:       - Name:            _start
# CHECK-2-NEXT:         Kind:            FUNCTION
# CHECK-2-NEXT:         Index:           0
# CHECK-2-NEXT:   - Type:            ELEM
# CHECK-2-NEXT:     Segments:
# CHECK-2-NEXT:       - Offset:
# CHECK-2-NEXT:           Opcode:          I32_CONST
# CHECK-2-NEXT:           Value:           1
# CHECK-2-NEXT:         Functions:       [ 1, 2, 0 ]
# CHECK-2-NEXT:   - Type:            CODE
# CHECK-2-NEXT:     Functions:
# CHECK-2-NEXT:       - Index:           0
# CHECK-2-NEXT:         Locals:          []
# CHECK-2-NEXT:         Body:            4181808080001A4182808080001A0B
# CHECK-2-NEXT:       - Index:           1
# CHECK-2-NEXT:         Locals:          []
# CHECK-2-NEXT:         Body:            0B
# CHECK-2-NEXT:       - Index:           2
# CHECK-2-NEXT:         Locals:          []
# CHECK-2-NEXT:         Body:            0B
# CHECK-2-NEXT:   - Type:            CUSTOM
# CHECK-2-NEXT:     Name:            name
# CHECK-2-NEXT:     FunctionNames:
# CHECK-2-NEXT:       - Index:           0
# CHECK-2-NEXT:         Name:            _start
# CHECK-2-NEXT:       - Index:           1
# CHECK-2-NEXT:         Name:            func1
# CHECK-2-NEXT:       - Index:           2
# CHECK-2-NEXT:         Name:            func2
# CHECK-2-NEXT:     GlobalNames:
# CHECK-2-NEXT:       - Index:           0
# CHECK-2-NEXT:         Name:            __stack_pointer

# CHECK-3:      --- !WASM
# CHECK-3-NEXT: FileHeader:
# CHECK-3-NEXT:   Version:         0x1
# CHECK-3-NEXT: Sections:
# CHECK-3-NEXT:   - Type:            TYPE
# CHECK-3-NEXT:     Signatures:
# CHECK-3-NEXT:       - Index:           0
# CHECK-3-NEXT:         ParamTypes:      []
# CHECK-3-NEXT:         ReturnTypes:     []
# CHECK-3-NEXT:   - Type:            FUNCTION
# CHECK-3-NEXT:     FunctionTypes:   [ 0, 0, 0 ]
# CHECK-3-NEXT:   - Type:            TABLE
# CHECK-3-NEXT:     Tables:
# CHECK-3-NEXT:       - Index:           0
# CHECK-3-NEXT:         ElemType:        FUNCREF
# CHECK-3-NEXT:         Limits:
# CHECK-3-NEXT:           Flags:           [ HAS_MAX ]
# CHECK-3-NEXT:           Minimum:         0x6
# CHECK-3-NEXT:           Maximum:         0x6
# CHECK-3-NEXT:   - Type:            MEMORY
# CHECK-3-NEXT:     Memories:
# CHECK-3-NEXT:       - Minimum:         0x2
# CHECK-3-NEXT:   - Type:            GLOBAL
# CHECK-3-NEXT:     Globals:
# CHECK-3-NEXT:       - Index:           0
# CHECK-3-NEXT:         Type:            I32
# CHECK-3-NEXT:         Mutable:         true
# CHECK-3-NEXT:         InitExpr:
# CHECK-3-NEXT:           Opcode:          I32_CONST
# CHECK-3-NEXT:           Value:           66560
# CHECK-3-NEXT:   - Type:            EXPORT
# CHECK-3-NEXT:     Exports:
# CHECK-3-NEXT:       - Name:            memory
# CHECK-3-NEXT:         Kind:            MEMORY
# CHECK-3-NEXT:         Index:           0
# CHECK-3-NEXT:       - Name:            _start
# CHECK-3-NEXT:         Kind:            FUNCTION
# CHECK-3-NEXT:         Index:           0
# CHECK-3-NEXT:   - Type:            ELEM
# CHECK-3-NEXT:     Segments:
# CHECK-3-NEXT:       - Offset:
# CHECK-3-NEXT:           Opcode:          I32_CONST
# CHECK-3-NEXT:           Value:           1
# CHECK-3-NEXT:         Functions:       [ 1, 0, 2, 0, 0 ]
# CHECK-3-NEXT:   - Type:            CODE
# CHECK-3-NEXT:     Functions:
# CHECK-3-NEXT:       - Index:           0
# CHECK-3-NEXT:         Locals:          []
# CHECK-3-NEXT:         Body:            4181808080001A4183808080001A0B
# CHECK-3-NEXT:       - Index:           1
# CHECK-3-NEXT:         Locals:          []
# CHECK-3-NEXT:         Body:            0B
# CHECK-3-NEXT:       - Index:           2
# CHECK-3-NEXT:         Locals:          []
# CHECK-3-NEXT:         Body:            0B
# CHECK-3-NEXT:   - Type:            CUSTOM
# CHECK-3-NEXT:     Name:            name
# CHECK-3-NEXT:     FunctionNames:
# CHECK-3-NEXT:       - Index:           0
# CHECK-3-NEXT:         Name:            _start
# CHECK-3-NEXT:       - Index:           1
# CHECK-3-NEXT:         Name:            func1
# CHECK-3-NEXT:       - Index:           2
# CHECK-3-NEXT:         Name:            func2
# CHECK-3-NEXT:     GlobalNames:
# CHECK-3-NEXT:       - Index:           0
# CHECK-3-NEXT:         Name:            __stack_pointer
