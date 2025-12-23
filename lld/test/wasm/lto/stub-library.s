## The function `bar` is declared in stub.so and depends on `foo` which is
## defined in an LTO object.  We also test the case where the LTO object is
## with an archive file.
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

## The function `foo` is declared in stub2.so and depends on `baz`, and both
## `foo` and `baz` are defined in an LTO object. When `foo` and `baz` are
## DCE'd and become undefined in the LTO process, wasm-ld should not try to
## export the (nonexistent) `baz`.

# RUN: llvm-mc -filetype=obj -triple=wasm32-unknown-unknown -o %t.o %s
# RUN: mkdir -p %t
# RUN: llvm-as %S/Inputs/foobaz.ll -o %t/foobaz.o
# RUN: wasm-ld %t.o %t/foobaz.o %p/Inputs/stub2.so -o %t.wasm
# RUN: obj2yaml %t.wasm | FileCheck %s --check-prefix=UNUSED

## Run the same test but with foobaz.o inside of an archive file.
# RUN: rm -f %t/libfoobaz.a
# RUN: llvm-ar rcs %t/libfoobaz.a %t/foobaz.o
# RUN: wasm-ld %t.o %t/libfoobaz.a %p/Inputs/stub2.so -o %t2.wasm
# RUN: obj2yaml %t2.wasm | FileCheck %s --check-prefix=UNUSED

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

# UNUSED:         Exports:
# UNUSED-NEXT:       - Name:            memory
# UNUSED-NEXT:         Kind:            MEMORY
# UNUSED-NEXT:         Index:           0
# UNUSED-NEXT:       - Name:            _start
# UNUSED-NEXT:         Kind:            FUNCTION
# UNUSED-NEXT:         Index:           1

# UNUSED-NOT:        - Name:            quux
