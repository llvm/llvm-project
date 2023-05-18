# REQUIRES: x86

# RUN: llvm-mc -filetype=obj -triple=x86_64 %p/Inputs/shared.s -o %t2.o
# RUN: ld.lld --hash-style=sysv -shared %t2.o -soname shared -o %t2.so
# RUN: llvm-mc -filetype=obj -triple=x86_64 %s -o %t.o

## Check exporting only one symbol.
# RUN: echo '{ foo1; };' > %t.list
# RUN: ld.lld --dynamic-list %t.list %t.o %t2.so -o %t
# RUN: llvm-nm -D %t | FileCheck %s --implicit-check-not=foo

## And now using quoted strings (the output is the same since it does not
## use any wildcard character).
# RUN: echo '{ "foo1"; };' > %t.list
# RUN: ld.lld --dynamic-list %t.list %t.o %t2.so -o %t2
# RUN: cmp %t %t2

## And now using --export-dynamic-symbol.
# RUN: ld.lld --export-dynamic-symbol foo1 %t.o %t2.so -o %t2
# RUN: cmp %t %t2
# RUN: ld.lld --export-dynamic-symbol=foo1 %t.o %t2.so -o %t2
# RUN: cmp %t %t2

# CHECK: foo1

## Now export all the foo1, foo2, and foo31 symbols
# RUN: echo "{ foo1; foo2; foo31; };" > %t.list
# RUN: ld.lld --dynamic-list %t.list %t.o %t2.so -o %t
# RUN: llvm-nm -D %t | FileCheck --check-prefix=CHECK2 %s --implicit-check-not=foo
# RUN: echo "{ foo1; foo2; };" > %t1.list
# RUN: echo "{ foo31; };" > %t2.list
# RUN: ld.lld --dynamic-list %t1.list --dynamic-list %t2.list %t.o %t2.so -o %t2
# RUN: cmp %t %t2

# CHECK2:      foo1
# CHECK2-NEXT: foo2
# CHECK2-NEXT: foo31

## --export-dynamic is similar to --dynamic-list with '{ * }'
# RUN: echo "{ foo2; };" > %t.list
# RUN: ld.lld --dynamic-list %t.list --export-dynamic %t.o %t2.so -o %t
# RUN: llvm-nm -D %t | FileCheck --check-prefix=CHECK3 %s --implicit-check-not=foo

## The same with --export-dynamic-symbol.
# RUN: ld.lld --export-dynamic-symbol=foo2 --export-dynamic %t.o %t2.so -o %t2
# RUN: cmp %t %t2

# CHECK3:      _start
# CHECK3-NEXT: foo1
# CHECK3-NEXT: foo2
# CHECK3-NEXT: foo31

.globl _start, foo1, foo2, foo31
foo1:
foo2:
foo31:
_start:
