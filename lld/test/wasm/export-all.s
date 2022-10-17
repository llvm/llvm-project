# RUN: llvm-mc -filetype=obj -triple=wasm32-unknown-unknown -o %t.o %s
# RUN: wasm-ld --export-all -o %t.wasm %t.o
# RUN: obj2yaml %t.wasm | FileCheck %s

.globaltype __stack_pointer, i32

.globl _start

_start:
  .functype _start () -> ()
  i32.const 3
  global.set __stack_pointer
  end_function

foo:
  .functype foo () -> (i32)
  i32.const 42
  end_function

#      CHECK:   - Type:            EXPORT
# CHECK-NEXT:     Exports:
# CHECK-NEXT:       - Name:            memory
# CHECK-NEXT:         Kind:            MEMORY
# CHECK-NEXT:         Index:           0
# CHECK-NEXT:       - Name:            __wasm_call_ctors
# CHECK-NEXT:         Kind:            FUNCTION
# CHECK-NEXT:         Index:           0
# CHECK-NEXT:       - Name:            _start
# CHECK-NEXT:         Kind:            FUNCTION
# CHECK-NEXT:         Index:           1
# CHECK-NEXT:       - Name:            __dso_handle
# CHECK-NEXT:         Kind:            GLOBAL
# CHECK-NEXT:         Index:           1
# CHECK-NEXT:       - Name:            __data_end
# CHECK-NEXT:         Kind:            GLOBAL
# CHECK-NEXT:         Index:           2
# CHECK-NEXT:       - Name:            __global_base
# CHECK-NEXT:         Kind:            GLOBAL
# CHECK-NEXT:         Index:           3
# CHECK-NEXT:       - Name:            __heap_base
# CHECK-NEXT:         Kind:            GLOBAL
# CHECK-NEXT:         Index:           4
# CHECK-NEXT:       - Name:            __heap_end
# CHECK-NEXT:         Kind:            GLOBAL
# CHECK-NEXT:         Index:           5
# CHECK-NEXT:       - Name:            __memory_base
# CHECK-NEXT:         Kind:            GLOBAL
# CHECK-NEXT:         Index:           6
# CHECK-NEXT:       - Name:            __table_base
# CHECK-NEXT:         Kind:            GLOBAL
# CHECK-NEXT:         Index:           7
