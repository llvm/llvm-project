# RUN: llvm-mc -filetype=obj -triple=wasm32-unknown-unknown %p/Inputs/ret32.s -o %t.ret32.o
# RUN: llvm-mc -filetype=obj -triple=wasm32-unknown-unknown -o %t.o %s
# RUN: wasm-ld -o %t.wasm %t.ret32.o %t.o
# RUN: obj2yaml %t.wasm | FileCheck %s

.functype ret32 (f32) -> (i32)

.globl  _start
_start:
  .functype _start () -> ()
  f32.const 0.000000e+00
  call ret32
  drop
  end_function

# CHECK:      Sections:
# CHECK:       - Type:            TYPE
# CHECK-NEXT:    Signatures:
# CHECK-NEXT:      - Index:           0
# CHECK-NEXT:        ParamTypes:
# CHECK-NEXT:          - F32
# CHECK-NEXT:        ReturnTypes:
# CHECK-NEXT:          - I32
# CHECK-NEXT:      - Index:           1
# CHECK-NEXT:        ParamTypes:
# CHECK-NEXT:        ReturnTypes:     []
# CHECK-NEXT:  - Type:            FUNCTION
# CHECK-NEXT:    FunctionTypes:   [ 0, 1 ]
# CHECK:       - Type:            CODE
# CHECK-NEXT:    Functions:
# CHECK:           - Index:       0
# CHECK:           - Index:       1
# CHECK:         Name:            name
# CHECK-NEXT:    FunctionNames:
# CHECK-NEXT:      - Index:           0
# CHECK-NEXT:        Name:            ret32
# CHECK-NEXT:      - Index:           1
# CHECK-NEXT:        Name:            _start
# CHECK-NEXT:    GlobalNames:
# CHECK-NEXT:      - Index:           0
# CHECK-NEXT:        Name:            __stack_pointer
# CHECK-NEXT: ...
