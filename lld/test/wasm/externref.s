# RUN: llvm-mc -triple=wasm32-unknown-unknown -filetype=obj -o %t.o %s
# RUN: wasm-ld %t.o -o %t.wasm
# RUN: obj2yaml %t.wasm | FileCheck %s

# Tests use of externref type, including storing it in a global

.globaltype my_global, externref

.globl read_externref
read_externref:
  .functype read_externref () -> (externref)
  global.get my_global
  end_function

.globl write_externref
write_externref:
  .functype write_externref (externref) -> ()
  local.get 0
  global.set my_global
  end_function

my_global:

.globl _start
_start:
  .functype _start () -> ()
  call read_externref
  call write_externref
  end_function

#      CHECK:  - Type:            GLOBAL
# CHECK-NEXT:    Globals:
# CHECK-NEXT:      - Index:           0
# CHECK-NEXT:        Type:            I32
# CHECK-NEXT:        Mutable:         true
# CHECK-NEXT:        InitExpr:
# CHECK-NEXT:          Opcode:          I32_CONST
# CHECK-NEXT:          Value:           66560
# CHECK-NEXT:      - Index:           1
# CHECK-NEXT:        Type:            EXTERNREF
# CHECK-NEXT:        Mutable:         true
# CHECK-NEXT:        InitExpr:
# CHECK-NEXT:          Opcode:          REF_NULL
# CHECK-NEXT:          Type:            EXTERNREF
