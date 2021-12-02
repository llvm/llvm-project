# Verify that we can handle R_WASM_FUNCTION_OFFSET relocations against live but
# undefined symbols.  Test that the .debug_info and .debug_int sections are
# generated without error
#
# Based on llvm/test/MC/WebAssembly/debuginfo-relocs.s
#
# RUN: llvm-mc -filetype=obj -triple=wasm32-unknown-unknown -o %t.o %s
# RUN: wasm-ld --import-undefined %t.o -o %t.wasm
# RUN: obj2yaml %t.wasm | FileCheck %s

.functype undef () -> ()

bar:
    .functype bar () -> ()
    end_function

    .globl _start
_start:
    .functype _start () -> ()
    call bar
    call undef
    end_function

.section .debug_int,"",@
.Ld:
  .int32 1
.size .Ld, 4

.section .debug_info,"",@
    .int32 bar
    .int32 undef
    .int32 .Ld

# CHECK:          Name:            .debug_info
# CHECK-NEXT:     Payload:         02000000FFFFFFFF00000000
# CHECK:          Name:            .debug_int
# CHECK-NEXT:     Payload:         '01000000'
