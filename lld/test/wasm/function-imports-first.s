# RUN: llvm-mc -filetype=obj -triple=wasm32-unknown-unknown %p/Inputs/ret32.s -o %t.ret32.o
# RUN: llvm-mc -filetype=obj -triple=wasm32-unknown-unknown %s -o %t.o
# RUN: wasm-ld -o %t.wasm %t.o %t.ret32.o
# RUN: obj2yaml %t.wasm | FileCheck %s

.functype ret32 (f32) -> (i32)

.globl _start
_start:
  .functype _start () -> ()
  f32.const 0.0
  call ret32
  drop
  end_function

# CHECK:  - Type:            TYPE
# CHECK:    Signatures:
# CHECK-NEXT:      - Index:           0
# CHECK-NEXT:        ParamTypes:      []
# CHECK-NEXT:        ReturnTypes:     []
# CHECK-NEXT:      - Index:           1
# CHECK-NEXT:        ParamTypes:
# CHECK-NEXT:          - F32
# CHECK-NEXT:        ReturnTypes:
# CHECK-NEXT:          - I32
# CHECK:  - Type:            FUNCTION
# CHECK-NEXT:    FunctionTypes: [ 0, 1 ]
# CHECK:  - Type:            CODE
# CHECK-NEXT:    Functions:
# CHECK-NEXT:      - Index:           0
# CHECK-NEXT:        Locals:
# CHECK-NEXT:        Body:            43000000001081808080001A0B
# CHECK-NEXT:      - Index:           1
# CHECK-NEXT:        Locals:
# CHECK-NEXT:        Body:            41000B
# CHECK-NEXT:   - Type:            CUSTOM
# CHECK-NEXT:     Name:            name
# CHECK-NEXT:     FunctionNames:
# CHECK-NEXT:       - Index:           0
# CHECK-NEXT:         Name:            _start
# CHECK-NEXT:       - Index:           1
# CHECK-NEXT:         Name:            ret32
# CHECK-NEXT: ...
