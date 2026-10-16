# RUN: llvm-mc -filetype=obj -triple=wasm32-unknown-unknown %s -o %t.o
# RUN: wasm-ld -wrap nosuchsym -wrap foo -allow-undefined -o %t.wasm %t.o
# RUN: obj2yaml %t.wasm | FileCheck %s

.globl foo
.globl _start

foo:
  .functype foo () -> ()
  end_function

_start:
  .functype _start () -> ()
  call  foo
  end_function

# CHECK:      - Type:            IMPORT
# CHECK-NEXT:   Imports:
# CHECK-NEXT:     - Module:          env
# CHECK-NEXT:       Field:           __wrap_foo
# CHECK-NEXT:       Kind:            FUNCTION
# CHECK-NEXT        SigIndex:        0

# CHECK:      - Type:            CODE
# CHECK-NEXT:   Functions:
# CHECK-NEXT:       Index:           1

# CHECK:        FunctionNames:
# CHECK-NEXT:      - Index:           0
# CHECK-NEXT:        Name:            __wrap_foo
# CHECK-NEXT:      - Index:           1
# CHECK-NEXT:        Name:            _start
