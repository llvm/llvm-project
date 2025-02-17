# RUN: llvm-mc -filetype=obj -triple=wasm32-unknown-unknown -o %t.o %s
# RUN: llvm-mc -filetype=obj -triple=wasm32-unknown-unknown %p/Inputs/ret32.s -o %t.ret32.o

# RUN: wasm-ld -shared --experimental-pic -o %t.ret32.so %t.ret32.o
# RUN: obj2yaml %t.ret32.so | FileCheck %s -check-prefix=SO1

# Without linking against the ret32.so shared object we expect an undefined
# symbol error

# RUN: not wasm-ld -shared --experimental-pic -o %t.so %t.o 2>&1 | FileCheck %s --check-prefix=ERROR
# ERROR: undefined symbol: ret32

# RUN: wasm-ld -shared --experimental-pic -o %t.so %t.o %t.ret32.so
# RUN: obj2yaml %t.so | FileCheck %s -check-prefix=SO2


.globl foo
.globl data

.functype ret32 (f32) -> (i32)

foo:
  .functype foo (f32) -> (i32)
  local.get 0
  call ret32
  end_function

.section .data,"",@
data:
 .p2align 2
 .int32 0
 .size data,4


# SO1:      Sections:
# SO1-NEXT:   - Type:            CUSTOM
# SO1-NEXT:     Name:            dylink.0
# SO1-NEXT:     MemorySize:      0
# SO1-NEXT:     MemoryAlignment: 0
# SO1-NEXT:     TableSize:       0
# SO1-NEXT:     TableAlignment:  0
# SO1-NEXT:     Needed:          []
# SO1-NEXT:   - Type:            TYPE

# SO2:      Sections:
# SO2-NEXT:   - Type:            CUSTOM
# SO2-NEXT:     Name:            dylink.0
# SO2-NEXT:     MemorySize:      4
# SO2-NEXT:     MemoryAlignment: 2
# SO2-NEXT:     TableSize:       0
# SO2-NEXT:     TableAlignment:  0
# SO2-NEXT:     Needed:
# SO2-NEXT:       - shared-needed.s.tmp.ret32.so
# SO2-NEXT:   - Type:            TYPE
