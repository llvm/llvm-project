# Test that linking without shared memory causes __tls_base to be
# internalized, and initialized to a non-zero value, even without any TLS data
# present.

# RUN: llvm-mc -filetype=obj -triple=wasm32-unknown-unknown -o %t.o %s
# RUN: wasm-ld -o %t.wasm %t.o
# RUN: obj2yaml %t.wasm | FileCheck %s

.globaltype __tls_base, i32

.globl _start
_start:
  .functype _start () -> ()
  global.get __tls_base
  drop
  end_function

# CHECK:        - Type:            GLOBAL
# CHECK-NEXT:     Globals:
# CHECK-NEXT:       - Index:           0
# CHECK-NEXT:         Type:            I32
# CHECK-NEXT:         Mutable:         true
# CHECK-NEXT:         InitExpr:
# CHECK-NEXT:           Opcode:          I32_CONST
# CHECK-NEXT:           Value:           65536
# CHECK-NEXT:       - Index:           1
# CHECK-NEXT:         Type:            I32
# CHECK-NEXT:         Mutable:         false
# CHECK-NEXT:         InitExpr:
# CHECK-NEXT:           Opcode:          I32_CONST
# CHECK-NEXT:           Value:           65536

# CHECK:          GlobalNames:
# CHECK-NEXT:       - Index:           0
# CHECK-NEXT:         Name:            __stack_pointer
# CHECK-NEXT:       - Index:           1
# CHECK-NEXT:         Name:            __tls_base
