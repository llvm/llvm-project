# REQUIRES: x86, shell

## The 'shell' requirement is to prevent this test from running by default on
## Windows as the extraction of the tar archive can cause problems related to
## path length limits.

# RUN: rm -rf %t.dir
# RUN: mkdir -p %t.dir
# RUN: cd %t.dir
# RUN: echo ".global foo; foo: ret" | \
# RUN:   llvm-mc -filetype=obj -triple=x86_64 - -o foo.o
# RUN: llvm-ar rc foo.a foo.o
# RUN: llvm-mc -filetype=obj -triple=x86_64 %s -o bar.o

# RUN: ld.lld bar.o -o /dev/null --reproduce repro.tar
# RUN: tar xf repro.tar
# RUN: cmp foo.a repro/%:t.dir/foo.a

.globl _start
_start:
  call foo
.section ".deplibs","MS",@llvm_dependent_libraries,1
.asciz "foo.a"
