# REQUIRES: x86
# RUN: llvm-mc -filetype=obj -triple=x86_64-unknown-linux %s -o %t

## Check simple RAM-only memory region.

# RUN: echo "MEMORY { ram (rwx) : ORIGIN = 0x8000, LENGTH = 256K } \
# RUN: SECTIONS { \
# RUN:   .text : { *(.text) } > ram \
# RUN:   .data : { *(.data) } > ram \
# RUN: }" > %t.script
# RUN: ld.lld -o %t1 --script %t.script %t
# RUN: llvm-readelf -S %t1 | FileCheck --check-prefix=RAM %s

# RAM:      [ 1] .text PROGBITS 0000000000008000 001000 000001
# RAM-NEXT: [ 2] .data PROGBITS 0000000000008001 001001 001000

## Check RAM and ROM memory regions.

# RUN: echo "MEMORY { \
# RUN:   ram (rwx) : ORIGIN = 0, LENGTH = 1024M \
# RUN:   rom (rx) : org = (0x80 * 0x1000 * 0x1000), len = 64M \
# RUN: } \
# RUN: SECTIONS { \
# RUN:   .text : { *(.text) } >rom \
# RUN:   .data : { *(.data) } >ram \
# RUN: }" > %t.script
# RUN: ld.lld -o %t1 --script %t.script %t
# RUN: llvm-readelf -S %t1 | FileCheck --check-prefix=RAMROM %s

# RAMROM:      [ 1] .text PROGBITS 0000000080000000 001000 000001
# RAMROM-NEXT: [ 2] .data PROGBITS 0000000000000000 002000 001000

## Check memory region placement by attributes.

# RUN: echo "MEMORY { \
# RUN:   ram (!rx) : ORIGIN = 0, LENGTH = 1024M \
# RUN:   rom (rx) : o = 0x80000000, l = 64M \
# RUN: } \
# RUN: SECTIONS { \
# RUN:   .text : { *(.text) } \
# RUN:   .data : { *(.data) } > ram \
# RUN: }" > %t.script
# RUN: ld.lld -o %t1 --script %t.script %t
# RUN: llvm-readelf -S %t1 | FileCheck --check-prefix=ATTRS %s

# ATTRS:      [ 1] .text PROGBITS 0000000080000000 001000 000001
# ATTRS-NEXT: [ 2] .data PROGBITS 0000000000000000 002000 001000

.text
.global _start
_start:
  nop

.data
b:
  .long 1
  .zero 4092
