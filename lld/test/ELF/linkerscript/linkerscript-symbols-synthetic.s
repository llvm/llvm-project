# REQUIRES: x86
# RUN: llvm-mc -filetype=obj -triple=x86_64-unknown-linux %s -o %t

# Simple symbol assignment within input section list. The '.' symbol
# is not location counter but offset from the beginning of output
# section .foo
# RUN: echo "SECTIONS { \
# RUN:          .foo : { \
# RUN:              begin_foo = .; \
# RUN:              *(.foo) \
# RUN:              end_foo = .; \
# RUN:              size_foo_1 = SIZEOF(.foo); \
# RUN:              . = ALIGN(0x1000); \
# RUN:              begin_bar = .; \
# RUN:              *(.bar) \
# RUN:              end_bar = .; \
# RUN:              size_foo_2 = SIZEOF(.foo); \ 
# RUN:            } \
# RUN:            size_foo_3 = SIZEOF(.foo); }" > %t.script
# RUN: ld.lld -o %t1 --script %t.script %t
# RUN: llvm-objdump -t %t1 | FileCheck --check-prefix=SIMPLE %s
# SIMPLE:      0000000000000120         .foo    00000000 begin_foo
# SIMPLE-NEXT: 0000000000000128         .foo    00000000 end_foo
# SIMPLE-NEXT: 0000000000000008         .foo    00000000 size_foo_1
# SIMPLE-NEXT: 0000000000001000         .foo    00000000 begin_bar
# SIMPLE-NEXT: 0000000000001004         .foo    00000000 end_bar
# SIMPLE-NEXT: 0000000000000ee4         .foo    00000000 size_foo_2
# SIMPLE-NEXT: 0000000000000ee4         *ABS*   00000000 size_foo_3

.global _start
_start:
 nop

.section .foo,"a"
 .quad 0

.section .bar,"a"
 .long 0
