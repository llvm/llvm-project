# REQUIRES: x86

# RUN: llvm-mc -filetype=obj -triple=x86_64 %s -o %t.o
# RUN: echo ".global foo; foo:" | \
# RUN:   llvm-mc -filetype=obj -triple=x86_64 - -o %tfoo.o
# RUN: rm -rf %t.dir
# RUN: mkdir -p %t.dir
# RUN: llvm-ar rc %t.dir/foo.a %tfoo.o
# RUN: not ld.lld %t.o -o /dev/null -L %t.dir 2>&1 | FileCheck %s -DOBJ=%t.o
# CHECK: error: [[OBJ]]: unable to find library from dependent library specifier: :foo.a

        .global _start
_start:
        call foo
    .section ".deplibs","MS",@llvm_dependent_libraries,1
        .asciz  ":foo.a"
