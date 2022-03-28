# REQUIRES: x86
## Test that COMMON matches common symbols.

# RUN: rm -rf %t && split-file %s %t
# RUN: llvm-mc -filetype=obj -triple=x86_64-unknown-linux %t/a.s -o %t/a.o
# RUN: ld.lld -T %t/1.t %t/a.o -o %t/a1
# RUN: llvm-readelf -S -s %t/a1 | FileCheck %s --check-prefix=CHECK1
# RUN: ld.lld -T %t/2.t %t/a.o -o %t/a2
# RUN: llvm-readelf -S -s %t/a2 | FileCheck %s --check-prefix=CHECK2

# CHECK1:      [Nr] Name    Type     Address  Off      Size   ES Flg Lk Inf Al
# CHECK1-NEXT: [ 0]         NULL     [[#%x,]] [[#%x,]] 000000 00      0   0  0
# CHECK1-NEXT: [ 1] .text   PROGBITS [[#%x,]] [[#%x,]] 000005 00  AX  0   0  4
# CHECK1-NEXT: [ 2] .data   PROGBITS [[#%x,]] [[#%x,]] 000001 00  WA  0   0  1
# CHECK1-NEXT: [ 3] .common NOBITS   [[#%x,]] [[#%x,]] 000180 00  WA  0   0 256
# CHECK1:         Value          Size Type    Bind   Vis     Ndx   Name
# CHECK1-DAG:  [[#%x,]]           128 OBJECT  GLOBAL DEFAULT [[#]] q1
# CHECK1-DAG:  [[#%x,]]           128 OBJECT  GLOBAL DEFAULT [[#]] q2

# CHECK2:      [Nr] Name    Type     Address  Off      Size   ES Flg Lk Inf Al
# CHECK2-NEXT: [ 0]         NULL     [[#%x,]] [[#%x,]] 000000 00      0   0  0
# CHECK2-NEXT: [ 1] .text   PROGBITS [[#%x,]] [[#%x,]] 000005 00  AX  0   0  4
# CHECK2-NEXT: [ 2] .data   PROGBITS [[#%x,]] [[#%x,]] 000180 00  WA  0   0 256
# CHECK2:         Value          Size Type    Bind   Vis     Ndx   Name
# CHECK2-DAG:  [[#%x,]]           128 OBJECT  GLOBAL DEFAULT [[#]] q1
# CHECK2-DAG:  [[#%x,]]           128 OBJECT  GLOBAL DEFAULT [[#]] q2

#--- a.s
.globl _start
_start:
  jmp _start

.section .data,"aw",@progbits
.byte 0

.comm q1,128,8
.comm q2,128,256

#--- 1.t
SECTIONS { . = SIZEOF_HEADERS; .common : { *(COMMON) } }

#--- 2.t
## COMMON can be placed in a SHT_PROGBITS section.
SECTIONS { . = SIZEOF_HEADERS; .data : { *(.data) *(COMMON) } }
