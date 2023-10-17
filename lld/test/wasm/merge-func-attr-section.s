# RUN: split-file %s %t
# RUN: llvm-mc -filetype=obj -triple=wasm32-unknown-unknown %t/a.s -o %t1.o
# RUN: llvm-mc -filetype=obj -triple=wasm32-unknown-unknown %t/b.s -o %t2.o
# RUN: wasm-ld -o %t.wasm %t1.o %t2.o
# RUN: obj2yaml %t.wasm | FileCheck %s

# Ensure two custom funct_attr sections are concatenated together.

# CHECK:       - Type:            CUSTOM
# CHECK-NEXT:    Name:            llvm.func_attr.custom0
# CHECK-NEXT:    Payload:         '000000000100000003000000'

#--- a.s

# Function index 3 (after linking)
        .functype       baz () -> ()
# Function index 1
        .functype       foo () -> ()
# Function index 2
        .functype       bar () -> ()
        .functype       _start () -> ()
        .globl  foo
        .type   foo,@function
foo:
        .functype       foo () -> ()
        end_function

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
        call    baz
        end_function

        .section        .custom_section.llvm.func_attr.custom0,"",@
        .int32  foo@FUNCINDEX
        .int32  bar@FUNCINDEX

#--- b.s
        .functype       baz () -> ()
        .globl  baz
        .type   baz,@function
baz:
        .functype       baz () -> ()
        end_function

        .section        .custom_section.llvm.func_attr.custom0,"",@
        .int32  baz@FUNCINDEX
