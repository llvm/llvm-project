# REQUIRES: x86
# UNSUPPORTED: system-windows
## Test that archive:file is supported in an input section description.

# RUN: mkdir -p %t.dir
# RUN: echo '.data; .byte 1' | llvm-mc -filetype=obj -triple=x86_64 - -o %t.dir/a.o
# RUN: llvm-mc -filetype=obj -triple=x86_64 %s -o %t.dir/b.o
# RUN: rm -f %t.a
# RUN: llvm-ar rc %t.a %t.dir/a.o %t.dir/b.o

## *.a:b.o matches /path/to/input-archive.s.tmp.a:b.o
## *b.o matches /path/to/input-archive.s.tmp.a:b.o
# RUN: echo 'SECTIONS { \
# RUN:   .foo : { "%t.a:a.o"(.data) } \
# RUN:   .bar : { *.a:b.o(.data) } \
# RUN:   .qux : { *b.o(.data1) } \
# RUN:   }' > %t.script
# RUN: ld.lld -T %t.script --whole-archive %t.a -o %t
# RUN: llvm-readelf -x .foo -x .bar -x .qux %t | FileCheck %s

# CHECK:      Hex dump of section '.foo':
# CHECK-NEXT: 0x00000000 01
# CHECK:      Hex dump of section '.bar':
# CHECK-NEXT: 0x00000001 02
# CHECK:      Hex dump of section '.qux':
# CHECK-NEXT: 0x00000002 03

.data
.byte 2

.section .data1,"aw",@progbits
.byte 3
