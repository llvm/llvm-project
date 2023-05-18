# RUN: llvm-mc -filetype=obj -triple=wasm32-unknown-unknown -o %t.o %s
# RUN: llvm-mc -filetype=obj -triple=wasm32-unknown-unknown %p/Inputs/ret32.s -o %t.ret32.o

# RUN: wasm-ld -shared --experimental-pic -o %t1.so %t.o
# RUN: obj2yaml %t1.so | FileCheck %s -check-prefix=SO1

# RUN: wasm-ld -shared --experimental-pic -o %t2.so %t1.so %t.ret32.o
# RUN: obj2yaml %t2.so | FileCheck %s -check-prefix=SO2

.globl foo
.globl data

foo:
  .functype foo () -> ()
  end_function

.section .data,"",@
data:
 .p2align 2
 .int32 0
 .size data,4


# SO1:      Sections:
# SO1-NEXT:   - Type:            CUSTOM
# SO1-NEXT:     Name:            dylink.0
# SO1-NEXT:     MemorySize:      4
# SO1-NEXT:     MemoryAlignment: 2
# SO1-NEXT:     TableSize:       0
# SO1-NEXT:     TableAlignment:  0
# SO1-NEXT:     Needed:          []
# SO1-NEXT:   - Type:            TYPE

# SO2:      Sections:
# SO2-NEXT:   - Type:            CUSTOM
# SO2-NEXT:     Name:            dylink.0
# SO2-NEXT:     MemorySize:      0
# SO2-NEXT:     MemoryAlignment: 0
# SO2-NEXT:     TableSize:       0
# SO2-NEXT:     TableAlignment:  0
# SO2-NEXT:     Needed:
# SO2-NEXT:       - shared-needed.s.tmp1.so
# SO2-NEXT:   - Type:            TYPE
