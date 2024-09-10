# RUN: llvm-mc -filetype=obj -triple=wasm32-unknown-unknown %s -o %t.o
# RUN: wasm-ld --export-all --import-table -o %t.wasm %t.o
# RUN: obj2yaml %t.wasm | FileCheck %s

.globl _start
_start:
  .functype _start () -> ()
  i32.const 1
  # call_indirect instruction implicitly references the function table
  call_indirect () -> ()
  end_function

# Verify the --import-table flag creates a table import

# CHECK:       - Type:            IMPORT
# CHECK-NEXT:    Imports:
# CHECK-NEXT:      - Module:          env
# CHECK-NEXT:        Field:           __indirect_function_table
# CHECK-NEXT:        Kind:            TABLE
# CHECK-NEXT:        Table:
# CHECK-NEXT:          Index:           0
# CHECK-NEXT:          ElemType:        FUNCREF
# CHECK-NEXT:          Limits:
# CHECK-NEXT:            Minimum:         0x1

