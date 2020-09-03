# REQUIRES: x86

# RUN: llvm-mc -filetype=obj -triple=x86_64 %s -o %t.o
# RUN: ld.lld --gc-sections --print-gc-sections %t.o -o /dev/null | FileCheck %s --implicit-check-not=removing

# CHECK: removing unused section {{.*}}.o:(.foo2)
# CHECK: removing unused section {{.*}}.o:(bar2)
# CHECK: removing unused section {{.*}}.o:(.zed2)

.global _start
_start:
.quad .foo1

.section .foo1,"a"
.quad 0

.section .foo2,"a"
.quad 0

.section bar1,"ao",@progbits,.foo1
.quad .zed1
.quad .foo1

.section bar2,"ao",@progbits,.foo2
.quad .zed2
.quad .foo2

.section .zed1,"a"
.quad 0

.section .zed2,"a"
.quad 0
