# REQUIRES: x86

# RUN: rm -rf %t.dir
# RUN: mkdir -p %t.dir
# RUN: cd %t.dir
# RUN: echo ".global foo; foo: ret" | \
# RUN:   llvm-mc -filetype=obj -triple=x86_64 - -o foo.o
# RUN: llvm-ar rc foo.a foo.o
# RUN: llvm-mc -filetype=obj -triple=x86_64 %s -o bar.o

# RUN: ld.lld bar.o -o /dev/null --reproduce repro.tar
# RUN: tar tf repro.tar | FileCheck -DPATH='%:t.dir' %s

# CHECK: [[PATH]]/foo.a

.globl _start
_start:
  call foo
.section ".deplibs","MS",@llvm_dependent_libraries,1
.asciz "foo.a"
