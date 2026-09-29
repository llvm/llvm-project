## The function `bar` is declared in stub.so and depends on `foo` which is
## defined in an LTO object.  We also test the case where the LTO object is
## with an archive file.
## The function `baz` is declared in stub.so and depends on `quux`, and both
## `baz` and `quux` are defined in an LTO object. When `baz` and `quux` are
## DCE'd and become undefined in the LTO process, wasm-ld should not try to
## export the (nonexistent) `quux`.
## This verifies that stub library dependencies (which are required exports) can
## be defined in LTO objects, even when they are within archive files.

# RUN: llvm-mc -filetype=obj -triple=wasm32-unknown-unknown -o %t.o %s
# RUN: mkdir -p %t
# RUN: llvm-as %S/Inputs/funcs.ll -o %t/funcs.o
# RUN: wasm-ld %t.o %t/funcs.o %p/Inputs/stub.so -o %t.wasm
# RUN: obj2yaml %t.wasm | FileCheck %s

## Run the same test but with funcs.o inside of an archive file.
# RUN: rm -f %t/libfuncs.a
# RUN: llvm-ar rcs %t/libfuncs.a %t/funcs.o
# RUN: wasm-ld %t.o %t/libfuncs.a %p/Inputs/stub.so -o %t2.wasm
# RUN: obj2yaml %t2.wasm | FileCheck %s

.functype bar () -> ()

.globl _start
_start:
    .functype _start () -> ()
    call bar
    end_function

# CHECK:         Imports:
# CHECK-NEXT:      - Module:          env
# CHECK-NEXT:        Field:           bar
# CHECK-NEXT:        Kind:            FUNCTION
# CHECK-NEXT:        SigIndex:        0

# CHECK:         Exports:
# CHECK-NEXT:       - Name:            memory
# CHECK-NEXT:         Kind:            MEMORY
# CHECK-NEXT:         Index:           0
# CHECK-NEXT:       - Name:            _start
# CHECK-NEXT:         Kind:            FUNCTION
# CHECK-NEXT:         Index:           1
# CHECK-NEXT:       - Name:            foo
# CHECK-NEXT:         Kind:            FUNCTION
# CHECK-NEXT:         Index:           2

# CHECK-NOT:        - Name:            quux
