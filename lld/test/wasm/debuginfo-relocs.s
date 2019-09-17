# RUN: llvm-mc -filetype=obj -triple=wasm32-unknown-unknown -o %t.o %s
# RUN: wasm-ld %t.o -o %t.wasm
# RUN: obj2yaml %t.wasm | FileCheck %s

bar:
    .functype bar () -> ()
    end_function

    .globl _start
_start:
    .functype _start () -> ()
    call bar
    end_function

    .section .debug_info,"",@
    .int32 bar

# Even though `bar` is live in the final binary it doesn't have a table entry
# since its not address taken in the code.  In this case any relocations in the
# debug sections see a address of zero.

# CHECK:         Name:            .debug_info
# CHECK-NEXT:    Payload:         '00000000'
