# RUN: llvm-mc -filetype=obj -triple=wasm32-unknown-unknown %p/Inputs/ret32.s -o %t.ret32.o
# RUN: llvm-mc -filetype=obj -triple=wasm32-unknown-unknown %s -o %t.main.o
# RUN: wasm-ld --export=ret32 -o %t.wasm %t.main.o %t.ret32.o
# RUN: obj2yaml %t.wasm | FileCheck %s

# Here we call ret32 with the wrong signature.  It actually returns void.
.functype ret32 (i32) -> (i32)

.globl _start
_start:
  .functype _start () -> ()
  i32.const 0
  call ret32
  drop
  end_function

# CHECK:        - Type:            EXPORT
# CHECK:            - Name:            ret32
# CHECK-NEXT:         Kind:            FUNCTION
# CHECK-NEXT:         Index:           2

# CHECK:        - Type:            CUSTOM
# CHECK-NEXT:     Name:            name
# CHECK-NEXT:     FunctionNames:
# CHECK-NEXT:       - Index:           0
# CHECK-NEXT:         Name:            'signature_mismatch:ret32'
# CHECK-NEXT:       - Index:           1
# CHECK-NEXT:         Name:            _start
# CHECK-NEXT:       - Index:           2
# CHECK-NEXT:         Name:            ret32
# CHECK-NEXT: ...
