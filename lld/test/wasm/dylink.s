# RUN: llvm-mc -filetype=obj -triple=wasm32-unknown-emscripten -o %t.o %s
# RUN: llvm-mc -filetype=obj -triple=wasm32-unknown-emscripten %p/Inputs/ret32.s -o %t.ret32.o
# RUN: llvm-mc -filetype=obj -triple=wasm32-unknown-emscripten %p/Inputs/libsearch-dyn.s -o %t.dyn.o
# RUN: wasm-ld --experimental-pic -shared %t.ret32.o %t.dyn.o -o %t.lib.so
# RUN: not wasm-ld --experimental-pic -pie -o %t.wasm %t.o 2>&1 | FileCheck --check-prefix=ERROR %s
# RUN: wasm-ld --experimental-pic -pie -o %t.wasm %t.o %t.lib.so
# RUN: obj2yaml %t.wasm | FileCheck %s

# Same again for wasm64

# RUN: llvm-mc -filetype=obj -triple=wasm64-unknown-emscripten -o %t.o %s
# RUN: llvm-mc -filetype=obj -triple=wasm64-unknown-emscripten %p/Inputs/ret32.s -o %t.ret32.o
# RUN: llvm-mc -filetype=obj -triple=wasm64-unknown-emscripten %p/Inputs/libsearch-dyn.s -o %t.dyn.o
# RUN: wasm-ld --experimental-pic -mwasm64 -shared %t.ret32.o %t.dyn.o -o %t.lib.so
# RUN: not wasm-ld --experimental-pic -mwasm64 -pie -o %t.wasm %t.o 2>&1 | FileCheck --check-prefix=ERROR %s
# RUN: wasm-ld --experimental-pic -mwasm64 -pie -o %t.wasm %t.o %t.lib.so
# RUN: obj2yaml %t.wasm | FileCheck %s

# ERROR: error: {{.*}}: undefined symbol: ret32
# ERROR: error: {{.*}}: undefined symbol: _bar
.functype ret32 (f32) -> (i32)

.globl _start
_start:
  .functype _start () -> ()
  f32.const 0.0
  call ret32
  drop
  i32.const _bar@GOT
  drop
  end_function

# CHECK:      Sections:
# CHECK-NEXT:   - Type:            CUSTOM
# CHECK-NEXT:     Name:            dylink.0
# CHECK-NEXT:     MemorySize:      0
# CHECK-NEXT:     MemoryAlignment: 0
# CHECK-NEXT:     TableSize:       0
# CHECK-NEXT:     TableAlignment:  0
# CHECK-NEXT:     Needed:
# CHECK-NEXT:       - {{.*}}.lib.so
