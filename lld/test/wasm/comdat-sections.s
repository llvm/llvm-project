# RUN: llvm-mc -triple=wasm32 -filetype=obj %p/Inputs/comdat1.s -o %t1.o
# RUN: llvm-mc -triple=wasm32 -filetype=obj %p/Inputs/comdat2.s -o %t2.o
# RUN: llvm-mc -triple=wasm32 -filetype=obj %s -o %t.o
# RUN: wasm-ld  -o %t.wasm %t.o %t1.o %t2.o
# RUN: obj2yaml %t.wasm | FileCheck %s


        .globl  _start
        .type  _start,@function
_start:
        .functype _start () -> ()
        call foo
        end_function

        .functype foo () -> ()


# Check that we got 1 copy of each of the .debug_foo sections from the 2 object
# files, and that they came from the same object.
# CHECK:  - Type:            CUSTOM
# CHECK-NEXT:    Name:            .debug_foo
# CHECK-NEXT:    Payload:         010000007B00000000000000
