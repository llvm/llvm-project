## The function `bar` is declared in stub.so and depends on `foo` which is
## defined in an LTO object.  We also test the case where the LTO object is
## with an archive file.
## The function `quux` is declared in stub.so and depends on `bar` which is
## defined in an LTO object as well. In this case, `bar` is DCE'd and becomes
## undefined in the LTO process, but `quux` should still be preserved and
## exported.
## This verifies that stub library dependencies (which are required exports) can
## be defined in LTO objects, even when they are within archive files.

# RUN: llvm-mc -filetype=obj -triple=wasm32-unknown-unknown -o %t.o %s
# RUN: mkdir -p %t
# RUN: llvm-as %S/Inputs/foo.ll -o %t/foo.o
# RUN: wasm-ld %t.o %t/foo.o %p/Inputs/stub.so -o %t.wasm
# RUN: obj2yaml %t.wasm | FileCheck %s

## Run the same test but with foo.o inside of an archive file.
# RUN: rm -f %t/libfoo.a
# RUN: llvm-ar rcs %t/libfoo.a %t/foo.o
# RUN: wasm-ld %t.o %t/libfoo.a %p/Inputs/stub.so -o %t2.wasm
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
# CHECK-NEXT:       - Name:            quux
# CHECK-NEXT:         Kind:            FUNCTION
# CHECK-NEXT:         Index:           3
