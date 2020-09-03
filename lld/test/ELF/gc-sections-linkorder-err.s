# REQUIRES: x86

## Error if the linked-to section of an input section is discarded.

# RUN: llvm-mc -filetype=obj -triple=x86_64 %s -o %t.o
# RUN: not ld.lld --gc-sections --print-gc-sections %t.o -o /dev/null 2>&1 | FileCheck %s

# CHECK:      removing unused section {{.*}}.o:(.foo0)
# CHECK-NEXT: error: {{.*}}.o:(.bar): sh_link points to discarded section {{.*}}.o:(.foo0)
# CHECK-NEXT: error: {{.*}}.o:(.baz): sh_link points to discarded section {{.*}}.o:(.foo0)

.globl _start
_start:
  call .foo1
  call bar0
  call bar1
  call baz0
  call baz1

.section .foo0,"a"
.section .foo1,"a"

## The linked-to section of the first input section is discarded.
.section .bar,"ao",@progbits,.foo0,unique,0
bar0:
.byte 0
.section .bar,"ao",@progbits,.foo1,unique,1
bar1:
.byte 1

## Another case: the linked-to section of the second input section is discarded.
.section .baz,"ao",@progbits,.foo1,unique,0
baz0:
.byte 0
.section .baz,"ao",@progbits,.foo0,unique,1
baz1:
.byte 1
