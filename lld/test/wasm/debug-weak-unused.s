# Verify that we can handle R_WASM_FUNCTION_OFFSET relocations against weak
# symbols that were defined in shared libraries.
# Test that the .debug_info is generated without error and contains the
# tombstone value.
#
# Based on debug-undefined-fs.s
#
# RUN: llvm-mc -filetype=obj -triple=wasm32-unknown-unknown -o %t_ret32.o %p/Inputs/ret32.s
# RUN: llvm-mc -filetype=obj -triple=wasm32-unknown-unknown -o %t.o %s
# RUN: wasm-ld -shared -o %t_lib.so %t_ret32.o
# RUN: wasm-ld -pie %t_lib.so %t.o -o %t.wasm
# RUN: obj2yaml %t.wasm | FileCheck %s

# ret32 is weakly defined here, but strongly defined in the shared library.
# So we expect the tombstone to be written for relocations refering to it in
# the debug section.
.globl ret32
.weak ret32
ret32:
    .functype ret32 (f32) -> (i32)
    i32.const 0
    end_function

.globl _start
_start:
    .functype _start () -> ()
    f32.const 0.0
    call ret32
    drop
    end_function

.section .debug_info,"",@
    .int32 ret32

# CHECK:          Name:            .debug_info
# CHECK-NEXT:     Payload:         FFFFFFFF
