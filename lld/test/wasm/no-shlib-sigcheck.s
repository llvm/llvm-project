# RUN: llvm-mc -filetype=obj -triple=wasm32-unknown-emscripten -o %t.o %s
# RUN: llvm-mc -filetype=obj -triple=wasm32-unknown-emscripten %p/Inputs/ret32.s -o %t.ret32.o
# RUN: wasm-ld --experimental-pic -shared %t.ret32.o -o %t.lib.so

## Fails with signature mismatch by default
# RUN: not wasm-ld --experimental-pic -pie -o %t.wasm %t.o %t.lib.so 2>&1 | FileCheck --check-prefix=ERROR %s
## Same again with shared library first.
# RUN: not wasm-ld --experimental-pic -pie -o %t.wasm %t.lib.so %t.o 2>&1 | FileCheck --check-prefix=ERROR %s

## Succeeds with --no-shlib-sigcheck added
# RUN: wasm-ld --experimental-pic -pie -o %t.wasm %t.o %t.lib.so --no-shlib-sigcheck
# RUN: obj2yaml %t.wasm | FileCheck %s
## Same again with shared library first.
# RUN: wasm-ld --experimental-pic -pie -o %t.wasm %t.lib.so %t.o --no-shlib-sigcheck
# RUN: obj2yaml %t.wasm | FileCheck %s

.functype ret32 (f32) -> (i64)

.globl _start
_start:
  .functype _start () -> ()
  f32.const 0.0
  call ret32
  drop
  end_function

# ERROR: wasm-ld: error: function signature mismatch: ret32
# ERROR: >>> defined as (f32) -> i64 in {{.*}}.o

# CHECK:       - Type:            TYPE
# CHECK-NEXT:    Signatures:
# CHECK-NEXT:      - Index:           0
# CHECK-NEXT:        ParamTypes:
# CHECK-NEXT:          - F32
# CHECK-NEXT:        ReturnTypes:
# CHECK-NEXT:          - I64
# CHECK-NEXT:      - Index:           1
# CHECK-NEXT:        ParamTypes:      []
# CHECK-NEXT:        ReturnTypes:     []
