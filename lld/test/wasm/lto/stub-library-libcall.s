# RUN: split-file %s %t
# RUN: llvm-mc -filetype=obj -triple=wasm32-unknown-unknown -o %t_main.o %t/main.s
# RUN: llvm-as %S/Inputs/foo.ll -o %t_foo.o
# RUN: llvm-as %S/Inputs/libcall.ll -o %t_libcall.o
# RUN: wasm-ld %t_main.o %t_libcall.o %t_foo.o %p/Inputs/stub.so -o %t.wasm
# RUN: obj2yaml %t.wasm | FileCheck %s

# The function `func_with_libcall` will generate an undefined reference to
# `memcpy` at LTO time.  `memcpy` itself also declared in stub.so and depends
# on `foo`

# If %t_foo.o is not included in the link we get an undefined symbol reported
# to the dependency of memcpy on the foo export:

# RUN: not wasm-ld %t_main.o %t_libcall.o %p/Inputs/stub.so -o %t.wasm 2>&1 | FileCheck --check-prefix=MISSING %s
# MISSING: stub.so: undefined symbol: foo. Required by memcpy

#--- main.s
.functype func_with_libcall (i32, i32) -> ()
.globl _start
_start:
    .functype _start () -> ()
    i32.const 0
    i32.const 0
    call func_with_libcall
    end_function

# CHECK:         Imports:
# CHECK-NEXT:      - Module:          env
# CHECK-NEXT:        Field:           memcpy
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
# CHECK-NEXT:         Index:           3
