# RUN: llvm-mc -filetype=obj -triple=wasm32-unknown-unknown -o %t.o %s
# RUN: wasm-ld %t.o -o %t.wasm
# RUN: obj2yaml %t.wasm | FileCheck %s

        .functype       foo () -> ()
        .functype       bar () -> ()
        .functype       _start () -> ()
        .globl  foo
        .type   foo,@function
foo:
        .functype       foo () -> ()
        end_function

        .section        .text.bar,"",@
        .globl  bar
        .type   bar,@function
bar:
        .functype       bar () -> ()
        end_function

        .globl  _start
        .type   _start,@function
_start:
        .functype       _start () -> ()
        call    foo
        call    bar
        end_function

        .section        .custom_section.llvm.func_attr.custom0,"",@
        .int32  foo@FUNCINDEX
        .int32  bar@FUNCINDEX

# CHECK:       - Type:            CUSTOM
# CHECK-NEXT:    Name:            llvm.func_attr.custom0
# CHECK-NEXT:    Payload:         '0000000001000000'
