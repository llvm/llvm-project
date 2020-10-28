# REQUIRES: x86
# RUN: llvm-mc -filetype=obj -triple=x86_64-pc-linux %s -o %t.o

## Contiguous SHF_LINK_ORDER sections.
# RUN: echo 'SECTIONS { .rodata : {BYTE(0) *(.rodata*) BYTE(4)} \
# RUN:   .text : {*(.text.bar) *(.text.foo)} }' > %t.lds
# RUN: ld.lld -T %t.lds %t.o -o %t
# RUN: llvm-readelf -S -x .rodata -x .text %t | FileCheck %s

# CHECK:      [ 1] .rodata   {{.*}} AL 3
# CHECK:      [ 3] .text     {{.*}} AX 0
# CHECK:      Hex dump of section '.rodata':
# CHECK-NEXT: 00030102 04
# CHECK:      Hex dump of section '.text':
# CHECK-NEXT: 0201

# RUN: echo 'SECTIONS { .rodata : {BYTE(0) *(.rodata*) BYTE(4)} \
# RUN:  .text : {*(.text.foo) *(.text.bar)} }' > %t1.lds
# RUN: ld.lld -T %t1.lds %t.o -o %t1
# RUN: llvm-readelf -S -x .rodata -x .text %t1 | FileCheck --check-prefix=CHECK1 %s

# CHECK1:      [ 1] .rodata   {{.*}} AL 3
# CHECK1:      [ 3] .text     {{.*}} AX 0
# CHECK1:      Hex dump of section '.rodata':
# CHECK1-NEXT: 00010302 04
# CHECK1:      Hex dump of section '.text':
# CHECK1-NEXT: 0102

## Adjacent input sections descriptions are contiguous.
## Orphan section .text.bar precedes .text.foo. However, don't swap the order of .rodata.*
## because they are in different InputSectionDescriptions.
# RUN: echo 'SECTIONS { .rodata : {*(.rodata.foo) *(.rodata.bar)} }' > %t2.lds
# RUN: ld.lld -T %t2.lds %t.o -o %t2
# RUN: llvm-readelf -S -x .rodata %t2 | FileCheck --check-prefix=CHECK2 %s

# CHECK2:      [ 1] .rodata   {{.*}} AL 5
# CHECK2:      [ 4] .text.bar {{.*}} AX 0
# CHECK2-NEXT: [ 5] .text.foo {{.*}} AX 0
# CHECK2:      Hex dump of section '.rodata':
# CHECK2-NEXT: 010302

## Non-contiguous SHF_LINK_ORDER sections, separated by a BYTE.
# RUN: echo 'SECTIONS { .rodata : {*(.rodata.foo) BYTE(0) *(.rodata.bar)} }' > %t3.lds
# RUN: ld.lld -T %t3.lds %t.o -o %t3
# RUN: llvm-readelf -S -x .rodata %t3 | FileCheck --check-prefix=CHECK3 %s

# CHECK3:      [ 1] .rodata   {{.*}} AL 5
# CHECK3:      [ 4] .text.bar {{.*}} AX 0
# CHECK3:      [ 5] .text.foo {{.*}} AX 0
# CHECK3:      Hex dump of section '.rodata':
# CHECK3-NEXT: 01000302

## Non-contiguous SHF_LINK_ORDER sections, separated by a non-SHF_LINK_ORDER section.
# RUN: echo 'SECTIONS { .rodata : {*(.rodata.foo) *(.text) *(.rodata.bar)} }' > %t4.lds
# RUN: ld.lld -T %t4.lds %t.o -o %t4
# RUN: llvm-readelf -x .rodata %t4 | FileCheck --check-prefix=CHECK4 %s

# CHECK4:      Hex dump of section '.rodata':
# CHECK4-NEXT: 01cccccc 0302

## Non-contiguous SHF_LINK_ORDER sections, separated by a symbol assignment.
# RUN: echo 'SECTIONS { .rodata : {*(.rodata.foo) a = .; *(.rodata.bar)} }' > %t5.lds
# RUN: ld.lld -T %t5.lds %t.o -o %t5
# RUN: llvm-readelf -S -x .rodata %t5 | FileCheck --check-prefix=CHECK2 %s

.global _start
_start:

.section .ro,"a"
.byte 0

.section .text.bar,"a",@progbits
.byte 2
.section .text.foo,"a",@progbits
.byte 1
.section .rodata.foo,"ao",@progbits,.text.foo
.byte 1
## If the two .rodata.bar sections are in the same InputSectionDescription,
## 03 (sh_link!=0) will be ordered before 02 (sh_link=0).
.section .rodata.bar,"a",@progbits
.byte 2
.section .rodata.bar,"ao",@progbits,.text.bar
.byte 3
