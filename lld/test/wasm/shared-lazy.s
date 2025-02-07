##  Based on lld/test/ELF/shared-lazy.s

# RUN: rm -rf %t && split-file %s %t && cd %t
# RUN: llvm-mc -filetype=obj -triple=wasm32-unknown-emscripten a.s -o a.o
# RUN: wasm-ld a.o --experimental-pic -shared -o a.so
# RUN: llvm-mc -filetype=obj -triple=wasm32-unknown-emscripten b.s -o b.o
# RUN: wasm-ld b.o --experimental-pic -shared -o b.so
# RUN: llvm-ar rc a.a a.o
# RUN: llvm-mc -filetype=obj -triple=wasm32-unknown-emscripten ref.s -o ref.o
# RUN: wasm-ld a.a b.so ref.o --experimental-pic -shared -o 1.so
# RUN: obj2yaml 1.so | FileCheck %s
# RUN: wasm-ld a.so a.a ref.o --experimental-pic -shared -o 1.so
# RUN: obj2yaml 1.so | FileCheck %s

## The definitions from a.so are used and we don't extract a member from the
## archive.

# CHECK:   - Type:            IMPORT
# CHECK:            - Module:          GOT.mem
# CHECK-NEXT:         Field:           x1
# CHECK-NEXT:         Kind:            GLOBAL
# CHECK-NEXT:         GlobalType:      I32
# CHECK-NEXT:         GlobalMutable:   true
# CHECK-NEXT:       - Module:          GOT.mem
# CHECK-NEXT:         Field:           x2
# CHECK-NEXT:         Kind:            GLOBAL
# CHECK-NEXT:         GlobalType:      I32
# CHECK-NEXT:         GlobalMutable:   true

## The extracted x1 is defined as STB_GLOBAL.
# RUN: wasm-ld ref.o a.a b.so -o 2.so --experimental-pic -shared
# RUN: obj2yaml 2.so | FileCheck %s --check-prefix=CHECK2
# RUN: wasm-ld a.a ref.o b.so -o 2.so --experimental-pic -shared
# RUN: obj2yaml 2.so | FileCheck %s --check-prefix=CHECK2

# CHECK2:       - Type:            EXPORT
# CHECK2-NEXT:    Exports:
# CHECK2-NEXT:      - Name:            __wasm_call_ctors
# CHECK2-NEXT:        Kind:            FUNCTION
# CHECK2-NEXT:        Index:
# CHECK2-NEXT:      - Name:            x1
# CHECK2-NEXT:        Kind:            GLOBAL
# CHECK2-NEXT:        Index:
# CHECK2-NEXT:      - Name:            x2
# CHECK2-NEXT:        Kind:            GLOBAL

#--- a.s
.section .data.x1,"",@
.global x1
x1:
  .byte 0
.size x1, 1

.section .data.x2,"",@
.weak x2
x2:
  .byte 0
.size x2, 1
#--- b.s
.section .data.x1,"",@
.globl x1
x1:
  .byte 0
.size x1, 1

.section .data.x2,"",@
.globl x2
x2:
  .byte 0
.size x2, 1
#--- ref.s
.globl x1
.globl x2
.globl d
.section .data.d,"",@
d:
  .int x1
  .int x2
.size d, 8
