# RUN: llvm-mc -filetype=obj -triple=wasm32-unknown-unknown %s -o %t.o

# RUN: wasm-ld -o %t.wasm %t.o
# RUN: obj2yaml %t.wasm | FileCheck %s  -check-prefix=CHECK-DEFAULT

# RUN: wasm-ld --function-pointer-alignment=2 -o %t.wasm %t.o
# RUN: obj2yaml %t.wasm | FileCheck %s  -check-prefix=CHECK-2

# RUN: wasm-ld --function-pointer-alignment=3 -o %t.wasm %t.o
# RUN: obj2yaml %t.wasm | FileCheck %s  -check-prefix=CHECK-3

.functype func1 () -> ()
.functype func2 () -> ()
.functype func3 () -> ()

.globl _start
_start:
  .functype _start () -> ()
  i32.const func1
  drop
  i32.const func2
  drop
  i32.const func3
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

.globl func3
func3:
  .functype func3 () -> ()
  end_function

# CHECK-DEFAULT:        - Type:            ELEM
# CHECK-DEFAULT-NEXT:     Segments:
# CHECK-DEFAULT-NEXT:       - Offset:
# CHECK-DEFAULT-NEXT:           Opcode:          I32_CONST
# CHECK-DEFAULT-NEXT:           Value:           1
# CHECK-DEFAULT-NEXT:         Functions:       [ 1, 2, 3 ]

# CHECK-2:        - Type:            ELEM
# CHECK-2-NEXT:     Segments:
# CHECK-2-NEXT:       - Offset:
# CHECK-2-NEXT:           Opcode:          I32_CONST
# CHECK-2-NEXT:           Value:           1
# CHECK-2-NEXT:         Functions:       [ 1, 2, 0, 3, 0 ]

# CHECK-3:        - Type:            ELEM
# CHECK-3-NEXT:     Segments:
# CHECK-3-NEXT:       - Offset:
# CHECK-3-NEXT:           Opcode:          I32_CONST
# CHECK-3-NEXT:           Value:           1
# CHECK-3-NEXT:         Functions:       [ 1, 0, 2, 0, 0, 3, 0, 0 ]
