# REQUIRES: x86
# RUN: llvm-mc -filetype=obj -triple=x86_64-pc-linux %s -o %t.o
# RUN: echo 'SECTIONS { \
# RUN:   . = SIZEOF_HEADERS; \
# RUN:   .text : { *(.text) } \
# RUN:   .tbss : { __tbss_start = .; *(.tbss) __tbss_end = .; } \
# RUN:   second_tbss : { second_tbss_start = .; *(second_tbss) second_tbss_end = .; } \
# RUN:   bar : { *(bar) } \
# RUN: }' > %t.lds
# RUN: ld.lld -T %t.lds %t.o -o %t1
# RUN: llvm-readelf -S -s %t1 | FileCheck %s

# RUN: echo 'PHDRS { text PT_LOAD; }' > %th.lds
# RUN: cat %th.lds %t.lds > %t2.lds
# RUN: ld.lld -T %t.lds %t.o -o %t2
# RUN: llvm-readelf -S -s %t2 | FileCheck %s

## Test that a tbss section doesn't affect the start address of the next section.

# CHECK: Name        Type     Address              Off                Size   ES Flg
# CHECK: .tbss       NOBITS   [[#%x,ADDR:]]        [[#%x,OFF:]]       000004 00 WAT
# CHECK: second_tbss NOBITS   {{0+}}[[#%x,ADDR+4]] {{0+}}[[#%x,OFF]]  000001 00 WAT
# CHECK: bar         PROGBITS {{0+}}[[#%x,ADDR]]   {{0+}}[[#%x,OFF]]  000004 00  WA

## Test that . in a tbss section represents the current location, even if the
## address will be reset.

# CHECK: Value                {{.*}} Name
# CHECK: {{0+}}[[#%x,ADDR]]   {{.*}} __tbss_start
# CHECK: {{0+}}[[#%x,ADDR+4]] {{.*}} __tbss_end
# CHECK: {{0+}}[[#%x,ADDR+4]] {{.*}} second_tbss_start
# CHECK: {{0+}}[[#%x,ADDR+5]] {{.*}} second_tbss_end

.globl _start
_start:
  nop

.section .tbss,"awT",@nobits
  .long 0
.section second_tbss,"awT",@nobits
  .byte 0
.section bar, "aw"
  .long 0
