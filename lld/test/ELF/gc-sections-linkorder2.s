# REQUIRES: x86

# RUN: llvm-mc -filetype=obj -triple=x86_64-pc-linux %s -o %t.o
# RUN: ld.lld --gc-sections --print-gc-sections %t.o -o /dev/null | count 0

.globl _start
_start:
.quad .foo

## .foo is retained, so sections linking to it are retained as well.
.section .foo,"a"
.quad 0
.section .bar,"ao",@progbits,.foo
.quad 0
.section .zed,"ao",@progbits,.foo
.quad 0

.section .nonalloc
.quad 0

.section .nonalloc_linkorder,"o",@progbits,.nonalloc
.quad 0
