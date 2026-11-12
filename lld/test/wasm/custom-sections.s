# RUN: llvm-mc -filetype=obj -triple=wasm32-unknown-unknown %s -o %t1.o
# RUN: llvm-mc -filetype=obj -triple=wasm32-unknown-unknown %S/Inputs/custom.s -o %t2.o
# RUN: wasm-ld --relocatable -o %t.wasm %t1.o %t2.o
# RUN: obj2yaml %t.wasm | FileCheck %s

.globl _start

_start:
  .functype _start () -> (i32)
  i32.const 0
  end_function

.section .custom_section.red,"",@
.ascii "extra"

# CHECK:        - Type:            CUSTOM
# CHECK-NEXT:     Name:            red
# CHECK-NEXT:     Payload:         6578747261666F6F
# CHECK-NEXT:   - Type:            CUSTOM
# CHECK-NEXT:     Name:            green
# CHECK-NEXT:     Payload:         '626172717578'
