# Test the --export of optional linker-synthetic symbols works.
# Specifically the __start_xxx and __end_xx symbols.

# RUN: llvm-mc -filetype=obj -triple=wasm32-unknown-unknown %s -o %t.o
# RUN: wasm-ld --export=__start_foo %t.o -o %t.wasm
# RUN: obj2yaml %t.wasm | FileCheck %s

  .globl  _start
_start:
  .functype _start () -> ()
  i32.const 0
  i32.load foo
  drop
  end_function

  .globl  foo
  .section    foo,"",@
foo:
  .int32 42
  .size foo, 4

#      CHECK:  - Type:            EXPORT
# CHECK-NEXT:    Exports:
# CHECK-NEXT:      - Name:            memory
# CHECK-NEXT:        Kind:            MEMORY
# CHECK-NEXT:        Index:           0
# CHECK-NEXT:      - Name:            _start
# CHECK-NEXT:        Kind:            FUNCTION
# CHECK-NEXT:        Index:           0
# CHECK-NEXT:      - Name:            __start_foo
# CHECK-NEXT:        Kind:            GLOBAL
# CHECK-NEXT:        Index:           1
