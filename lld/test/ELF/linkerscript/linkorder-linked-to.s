# REQUIRES: x86
## Test how we set the sh_link field of a SHF_LINK_ORDER output section.
## Additionally, test that in a relocatable link, SHF_LINK_ORDER sections are
## not combined. Combining them will arbitrarily choose a single output
## section, losing tracking of correct dependencies.

# RUN: llvm-mc -filetype=obj --triple=x86_64 %s -o %t.o

## In the absence of a SECTIONS command.
# RUN: ld.lld %t.o -o %t
# RUN: llvm-readelf -S -x foo %t | FileCheck --check-prefixes=EXE,EXE-HEX %s
# RUN: ld.lld %t.o -o %t.ro -r
# RUN: llvm-readelf -S %t.ro | FileCheck --check-prefix=REL %s

## A non-relocatable link places readonly sections before read-executable sections.
# EXE:      [Nr] Name     {{.*}} Flg Lk
# EXE-NEXT: [ 0]
# EXE-NEXT: [ 1] foo      {{.*}}  AL  2
# EXE-NEXT: [ 2] .text    {{.*}}  AX  0

## An implementation detail: foo is moved after its linked-to section (orphan).
# REL:      [Nr] Name     {{.*}} Flg Lk
# REL-NEXT: [ 0]
# REL-NEXT: [ 1] .text    {{.*}}  AX  0
# REL-NEXT: [ 2] .text.f1 {{.*}}  AX  0
# REL-NEXT: [ 3] foo      {{.*}}  AL  2
# REL-NEXT: [ 4] .text.f2 {{.*}}  AX  0
# REL-NEXT: [ 5] foo      {{.*}}  AL  4

## A SECTIONS command combines .text.*
# RUN: echo 'SECTIONS { .text : { *(.text.f1) *(.text.f2) } }' > %t1.lds
# RUN: ld.lld -T %t1.lds %t.o -o %t1
# RUN: llvm-readelf -S -x foo %t1 | FileCheck --check-prefixes=EXE,EXE-HEX %s
# RUN: ld.lld -T %t1.lds %t.o -o %t1.ro -r
# RUN: llvm-readelf -S -x foo %t1.ro | FileCheck --check-prefix=REL1 %s

# REL1:      [Nr] Name    {{.*}} Flg Lk
# REL1-NEXT: [ 0]
# REL1-NEXT: [ 1] .text   {{.*}}  AX  0
# REL1-NEXT: [ 2] foo     {{.*}}  AL  1

## A SECTIONS command separates .text.*
# RUN: echo 'SECTIONS { .text.f1 : { *(.text.f1) } .text.f2 : { *(.text.f2) } }' > %t2.lds
# RUN: ld.lld -T %t2.lds %t.o -o %t2
# RUN: llvm-readelf -S -x foo %t2 | FileCheck --check-prefixes=EXE2,EXE-HEX %s
# RUN: ld.lld -T %t2.lds %t.o -o %t2.ro -r
# RUN: llvm-readelf -S %t2.ro | FileCheck --check-prefixes=REL2 %s

# EXE2:      [Nr] Name     {{.*}} Flg Lk
# EXE2-NEXT: [ 0]
# EXE2-NEXT: [ 1] foo      {{.*}}  AL  2
# EXE2-NEXT: [ 2] .text.f1 {{.*}}  AX  0
# EXE2-NEXT: [ 3] .text.f2 {{.*}}  AX  0

# REL2:      [Nr] Name     {{.*}} Flg Lk
# REL2-NEXT: [ 0]
# REL2-NEXT: [ 1] .text.f1 {{.*}}  AX  0
# REL2-NEXT: [ 2] .text.f2 {{.*}}  AX  0
# REL2-NEXT: [ 3] .text    {{.*}}  AX  0
# REL2-NEXT: [ 4] foo      {{.*}}  AL  1
# REL2-NEXT: [ 5] foo      {{.*}}  AL  2

# EXE-HEX:      Hex dump of section 'foo':
# EXE-HEX-NEXT: 0102

.section .text.f1,"ax",@progbits
ret
.section .text.f2,"ax",@progbits
ret

.section foo,"ao",@progbits,.text.f2
.byte 2
.section foo,"ao",@progbits,.text.f1
.byte 1
