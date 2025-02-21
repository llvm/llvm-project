## Check that synthetic data-layout symbols such as __heap_base and __heap_end
## can be referenced from shared libraries and pie executables without
## generating undefined symbols.

# RUN: llvm-mc -filetype=obj -triple=wasm32-unknown-unknown -o %t.o %s
# RUN: wasm-ld --experimental-pic -pie --import-memory -o %t.wasm %t.o
# RUN: obj2yaml %t.wasm | FileCheck %s
# RUN: wasm-ld --experimental-pic -shared -o %t.so %t.o
# RUN: obj2yaml %t.so | FileCheck %s

.globl _start

_start:
  .functype _start () -> ()
  i32.const __heap_base@GOT
  drop
  i32.const __heap_end@GOT
  drop
  i32.const __stack_low@GOT
  drop
  i32.const __stack_high@GOT
  drop
  i32.const __global_base@GOT
  drop
  i32.const __data_end@GOT
  drop
  end_function

# CHECK:        - Type:            IMPORT
# CHECK-NEXT:     Imports:
# CHECK-NEXT:       - Module:          env
# CHECK-NEXT:         Field:           memory
# CHECK-NEXT:         Kind:            MEMORY
# CHECK-NEXT:         Memory:
# CHECK-NEXT:           Minimum:         0x0
# CHECK-NEXT:       - Module:          env
# CHECK-NEXT:         Field:           __memory_base
# CHECK-NEXT:         Kind:            GLOBAL
# CHECK-NEXT:         GlobalType:      I32
# CHECK-NEXT:         GlobalMutable:   false
# CHECK-NEXT:       - Module:          env
# CHECK-NEXT:         Field:           __table_base
# CHECK-NEXT:         Kind:            GLOBAL
# CHECK-NEXT:         GlobalType:      I32
# CHECK-NEXT:         GlobalMutable:   false
# CHECK-NEXT:       - Module:          GOT.mem
# CHECK-NEXT:         Field:           __heap_base
# CHECK-NEXT:         Kind:            GLOBAL
# CHECK-NEXT:         GlobalType:      I32
# CHECK-NEXT:         GlobalMutable:   true
# CHECK-NEXT:       - Module:          GOT.mem
# CHECK-NEXT:         Field:           __heap_end
# CHECK-NEXT:         Kind:            GLOBAL
# CHECK-NEXT:         GlobalType:      I32
# CHECK-NEXT:         GlobalMutable:   true
# CHECK-NEXT:       - Module:          GOT.mem
# CHECK-NEXT:         Field:           __stack_low
# CHECK-NEXT:         Kind:            GLOBAL
# CHECK-NEXT:         GlobalType:      I32
# CHECK-NEXT:         GlobalMutable:   true
# CHECK-NEXT:       - Module:          GOT.mem
# CHECK-NEXT:         Field:           __stack_high
# CHECK-NEXT:         Kind:            GLOBAL
# CHECK-NEXT:         GlobalType:      I32
# CHECK-NEXT:         GlobalMutable:   true
# CHECK-NEXT:       - Module:          GOT.mem
# CHECK-NEXT:         Field:           __global_base
# CHECK-NEXT:         Kind:            GLOBAL
# CHECK-NEXT:         GlobalType:      I32
# CHECK-NEXT:         GlobalMutable:   true
# CHECK-NEXT:       - Module:          GOT.mem
# CHECK-NEXT:         Field:           __data_end
# CHECK-NEXT:         Kind:            GLOBAL
# CHECK-NEXT:         GlobalType:      I32
# CHECK-NEXT:         GlobalMutable:   true
