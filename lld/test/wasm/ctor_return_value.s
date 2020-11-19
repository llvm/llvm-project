# RUN: llvm-mc -filetype=obj -triple=wasm32-unknown-unknown -o %t.o %s
# RUN: wasm-ld %t.o -o %t.wasm
# RUN: obj2yaml %t.wasm | FileCheck %s

  .globl  myctor
myctor:
  .functype myctor () -> (i32)
  i32.const 1

  end_function

  .globl  _start
_start:
  .functype _start () -> ()
  call __wasm_call_ctors
  end_function

  .section  .init_array.100,"",@
  .p2align  2
  .int32  myctor
  .int32  myctor
  .int32  myctor

.type __wasm_call_ctors,@function

#      CHECK:   - Type:            CODE
# CHECK-NEXT:     Functions:
# CHECK-NEXT:       - Index:           0
# CHECK-NEXT:         Locals:          []
# CHECK-NEXT:         Body:            10011A10011A10011A0B
# CHECK-NEXT:       - Index:           1
# CHECK-NEXT:         Locals:          []
# CHECK-NEXT:         Body:            41010B
# CHECK-NEXT:       - Index:           2
# CHECK-NEXT:         Locals:          []
# CHECK-NEXT:         Body:            1080808080000B
# CHECK-NEXT:   - Type:            CUSTOM
# CHECK-NEXT:     Name:            name
# CHECK-NEXT:     FunctionNames:
# CHECK-NEXT:       - Index:           0
# CHECK-NEXT:         Name:            __wasm_call_ctors
# CHECK-NEXT:       - Index:           1
# CHECK-NEXT:         Name:            myctor
# CHECK-NEXT:       - Index:           2
# CHECK-NEXT:         Name:            _start
# CHECK-NEXT:     GlobalNames:
# CHECK-NEXT:       - Index:           0
# CHECK-NEXT:         Name:            __stack_pointer
# CHECK-NEXT: ...
