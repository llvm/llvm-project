# RUN: llvm-mc -filetype=obj -triple=wasm32-unknown-unknown -o %t.o %s
# RUN: wasm-ld -shared -o %t1.wasm %t.o -rpath /a/b/c -rpath /x/y/z --experimental-pic
# RUN: obj2yaml %t1.wasm | FileCheck %s

# CHECK:  - Type:            CUSTOM
# CHECK-NEXT:    Name:            dylink.0
# CHECK-NEXT:    MemorySize:      0
# CHECK-NEXT:    MemoryAlignment: 0
# CHECK-NEXT:    TableSize:       0
# CHECK-NEXT:    TableAlignment:  0
# CHECK-NEXT:    Needed:          []
# CHECK-NEXT:    RuntimePath:
# CHECK-NEXT:      - '/a/b/c'
# CHECK-NEXT:      - '/x/y/z'
