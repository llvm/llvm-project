# RUN: llvm-mc -filetype=obj -triple=wasm32-unknown-unknown -o %t.o %s
# RUN: llvm-mc -filetype=obj -triple=wasm32-unknown-unknown -o %t2.o %S/Inputs/sig-mismatch.s
# RUN: wasm-ld --relocatable %t.o %t2.o -o %t.wasm
# RUN: obj2yaml %t.wasm | FileCheck %s

# Regression test for handling of signature mismatches (variant function
# symbols) and relocatable output.  This issue only occurred when the undefined
# function was seen first and the defined function was referenced within the
# the defining file (see %S/Inputs/sig-mismatch.s).

.functype foo (i32, i64, i32) -> (i32)

.globl _start
_start:
  .functype _start () -> ()
  i32.const 1
  i64.const 2
  i32.const 3
  call foo
  drop
  end_function

#      CHECK:  - Type:            CUSTOM
# CHECK-NEXT:    Name:            linking
# CHECK-NEXT:    Version:         2
# CHECK-NEXT:    SymbolTable:
# CHECK-NEXT:      - Index:           0
# CHECK-NEXT:        Kind:            FUNCTION
# CHECK-NEXT:        Name:            _start
# CHECK-NEXT:        Flags:           [  ]
# CHECK-NEXT:        Function:        1
# CHECK-NEXT:      - Index:           1
# CHECK-NEXT:        Kind:            FUNCTION
# CHECK-NEXT:        Name:            foo
# CHECK-NEXT:        Flags:           [  ]
# CHECK-NEXT:        Function:        2
# CHECK-NEXT:      - Index:           2
# CHECK-NEXT:        Kind:            FUNCTION
# CHECK-NEXT:        Name:            call_foo
# CHECK-NEXT:        Flags:           [  ]
# CHECK-NEXT:        Function:        3
# CHECK-NEXT:      - Index:           3
# CHECK-NEXT:        Kind:            FUNCTION
# CHECK-NEXT:        Name:            'signature_mismatch:foo'
# CHECK-NEXT:        Flags:           [ BINDING_LOCAL ]
# CHECK-NEXT:        Function:        0
# CHECK-NEXT:  - Type:            CUSTOM
# CHECK-NEXT:    Name:            name
# CHECK-NEXT:    FunctionNames:
# CHECK-NEXT:      - Index:           0
# CHECK-NEXT:        Name:            'signature_mismatch:foo'
# CHECK-NEXT:      - Index:           1
# CHECK-NEXT:        Name:            _start
# CHECK-NEXT:      - Index:           2
# CHECK-NEXT:        Name:            foo
# CHECK-NEXT:      - Index:           3
# CHECK-NEXT:        Name:            call_foo
# CHECK-NEXT:...
