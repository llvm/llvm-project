# REQUIRES: x86
## A zero-sized section following .bss must not account for .bss's size.

# RUN: llvm-mc -filetype=obj -triple=x86_64 %s -o %t.o

# RUN: echo 'SECTIONS { \
# RUN:   .text : {} \
# RUN:   .data : {} \
# RUN:   .bss : {} \
# RUN:   .ldata : { . = .; } \
# RUN: }' > %t-a.lds
# RUN: ld.lld -T %t-a.lds %t.o -o %t-a
# RUN: llvm-readelf -S %t-a | FileCheck --check-prefix=ZERO-SIZE %s

# RUN: echo 'SECTIONS { \
# RUN:   .text : {} \
# RUN:   .data : {} \
# RUN:   .bss : {} \
# RUN:   .ldata : { . = ALIGN(8); } \
# RUN: }' > %t-b.lds
# RUN: ld.lld -T %t-b.lds %t.o -o %t-b
# RUN: llvm-readelf -S %t-b | FileCheck --check-prefix=ALIGN-ZERO-SIZE %s

## The .ldata section's file offset must equal .bss's file offset.
# ZERO-SIZE: .bss    NOBITS   {{[0-9a-f]+}} [[OFF:[0-9a-f]+]] {{[0-9a-f]+}}
# ZERO-SIZE-NEXT: .ldata  PROGBITS {{[0-9a-f]+}} [[OFF]] 000000

# ALIGN-ZERO-SIZE: .bss    NOBITS   {{[0-9a-f]+}} [[OFF:[0-9a-f]+]] {{[0-9a-f]+}}
# ALIGN-ZERO-SIZE-NEXT: .ldata  PROGBITS {{[0-9a-f]+}} [[OFF]] 000000

.text
.globl _start
_start:
  ret

.data
.byte 1

.bss
.space 0xfa6
