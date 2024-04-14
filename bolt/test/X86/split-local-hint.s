## Test the local symbol name hint is used when a local function is split.

# RUN: split-file %s %t
# RUN: llvm-mc --filetype=obj --triple x86_64-unknown-unknown %t/global -o %t.global.o
# RUN: llvm-mc --filetype=obj --triple x86_64-unknown-unknown %t/local -o %t.local.o
# RUN: llvm-mc --filetype=obj --triple x86_64-unknown-unknown %t/local2 -o %t.local2.o
# RUN: ld.lld %t.global.o %t.local2.o %t.local.o -o %t.exe -q
# RUN: llvm-nm %t.exe  | FileCheck %s --check-prefix=CHECK-NM
# RUN: llvm-bolt %t.exe -o %t.bolt --split-functions --split-strategy=random2 \
# RUN:   -lite=0 --bolt-seed=7 --enable-bat
# RUN: llvm-nm %t.bolt | FileCheck %s --check-prefix=CHECK-NM-BOLT
# RUN: link_fdata %s %t.bolt %t.preagg PREAGG
# PREAGG: B X:0 #main# 1 0
# RUN: perf2bolt %t.bolt -p %t.preagg --pa -v=1 -o %t.null \
# RUN:   | FileCheck %s --check-prefix=CHECK-BOLT

# CHECK-NM-DAG:  T foo
# CHECK-NM-DAG:  t foo
# CHECK-NM-DAG:  t foo
# CHECK-NM-BOLT: t foo.cold_[[#]].0
# CHECK-BOLT: BOLT-INFO: marking foo.cold_[[#]].0/1(*2) as a fragment of foo/2

#--- global
.text
.file "global.s"
        .globl  foo
        .type   foo, @function
foo:
        ud2
        .size   foo, .-foo

#--- local2
.text
.file "local2.s"
        .type   foo, @function
foo:
        ret
        .size   foo, .-foo

#--- local
.file "local.s"
.text
        .type   foo, @function
foo:
.L3:
        subl    $1, %edi
        testl   %edi, %edi
        jg      .L3
        jmp     .L4
.L4:
        subl    $1, %edi
        subl    $1, %edi
        subl    $1, %edi
        subl    $1, %edi
        ret
        .size   foo, .-foo


        .globl  main
        .type   main, @function
main:
        movl    $10, %edi
        call    foo
        ret
        .size   main, .-main
