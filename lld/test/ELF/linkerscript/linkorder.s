# REQUIRES: x86
# RUN: llvm-mc -filetype=obj -triple=x86_64-pc-linux %s -o %t.o

## Contiguous SHF_LINK_ORDER sections.
# RUN: echo 'SECTIONS { .rodata : {BYTE(0) *(.rodata*) BYTE(3)} \
# RUN:   .text : {*(.text.bar) *(.text.foo)} }' > %t.lds
# RUN: ld.lld -T %t.lds %t.o -o %t
# RUN: llvm-readelf -S -x .rodata -x .text %t | FileCheck %s

# CHECK:      [ 1] .rodata   {{.*}} AL 3
# CHECK:      [ 3] .text     {{.*}} AX 0
# CHECK:      Hex dump of section '.rodata':
# CHECK-NEXT: 00020103
# CHECK:      Hex dump of section '.text':
# CHECK-NEXT: 0201

# RUN: echo 'SECTIONS { .rodata : {BYTE(0) *(.rodata*) BYTE(3)} \
# RUN:  .text : {*(.text.foo) *(.text.bar)} }' > %t1.lds
# RUN: ld.lld -T %t1.lds %t.o -o %t1
# RUN: llvm-readelf -S -x .rodata -x .text %t1 | FileCheck --check-prefix=CHECK1 %s

# CHECK1:      [ 1] .rodata   {{.*}} AL 3
# CHECK1:      [ 3] .text     {{.*}} AX 0
# CHECK1:      Hex dump of section '.rodata':
# CHECK1-NEXT: 00010203
# CHECK1:      Hex dump of section '.text':
# CHECK1-NEXT: 0102

## Adjacent input sections descriptions are contiguous.
## Orphan section .text.bar precedes .text.foo, so swap the order of .rodata.*
# RUN: echo 'SECTIONS { .rodata : {*(.rodata.foo) *(.rodata.bar)} }' > %t2.lds
# RUN: ld.lld -T %t2.lds %t.o -o %t2
# RUN: llvm-readelf -S -x .rodata %t2 | FileCheck --check-prefix=CHECK2 %s

# CHECK2:      [ 1] .rodata   {{.*}} AL 3
# CHECK2:      [ 3] .text     {{.*}} AX 0
# CHECK2:      Hex dump of section '.rodata':
# CHECK2-NEXT: 0201

## Non-contiguous SHF_LINK_ORDER sections, separated by a BYTE.
# RUN: echo 'SECTIONS { .rodata : {*(.rodata.foo) BYTE(0) *(.rodata.bar)} }' > %terr1.lds
# RUN: not ld.lld -T %terr1.lds %t.o -o /dev/null 2>&1 | FileCheck --check-prefix=ERR %s

## Non-contiguous SHF_LINK_ORDER sections, separated by a non-SHF_LINK_ORDER section.
# RUN: echo 'SECTIONS { .rodata : {*(.rodata.foo) *(.text) *(.rodata.bar)} }' > %terr2.lds
# RUN: not ld.lld -T %terr2.lds %t.o -o /dev/null 2>&1 | FileCheck --check-prefix=ERR %s

## Non-contiguous SHF_LINK_ORDER sections, separated by a symbol assignment.
# RUN: echo 'SECTIONS { .rodata : {*(.rodata.foo) a = .; *(.rodata.bar)} }' > %terr3.lds
# RUN: not ld.lld -T %terr3.lds %t.o -o /dev/null 2>&1 | FileCheck --check-prefix=ERR %s

# ERR: error: {{.*}}.o:(.rodata.bar): SHF_LINK_ORDER sections in .rodata are not contiguous

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
.section .rodata.bar,"ao",@progbits,.text.bar
.byte 2
