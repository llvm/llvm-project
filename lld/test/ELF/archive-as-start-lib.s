# REQUIRES: x86
# UNSUPPORTED: system-windows

# RUN: rm -rf %t && split-file %s %t && cd %t
# RUN: llvm-mc -filetype=obj -triple=x86_64 a.s -o a.o
# RUN: llvm-mc -filetype=obj -triple=x86_64 b.s -o b.o

## Create an archive with incomplete index: foo is missing.
# RUN: llvm-ar --format=gnu rc a.a a.o
# RUN: llvm-ar --format=gnu rcS b.a b.o && tail -c +9 b.a > b-tail
# RUN: cat a.a b-tail > weird.a
# RUN: llvm-nm --print-armap weird.a | FileCheck %s --check-prefix=ARMAP

# ARMAP:      Archive map
# ARMAP-NEXT: _start in a.o
# ARMAP-EMPTY:

## The incomplete archive index is ignored. -u foo extracts weird.a(b.o).
## In GNU ld, foo is undefined.
# RUN: ld.lld -m elf_x86_64 -u foo weird.a -o lazy
# RUN: llvm-nm lazy | FileCheck %s --implicit-check-not={{.}}

# CHECK: [[#%x,]] T _start
# CHECK: [[#%x,]] T foo

#--- a.s
.globl _start
_start:

#--- b.s
.globl foo
foo:
