# REQUIRES: x86
# RUN: llvm-mc -filetype=obj -triple=x86_64-pc-linux %s -o %t.o
# RUN: echo "SECTIONS { \
# RUN:         .sec1 (NOLOAD) : { . += 1; } \
# RUN:         .bss           : { *(.bss) } \
# RUN:       };" > %t.script
# RUN: ld.lld %t.o -T %t.script -o %t
# RUN: llvm-readelf -S -l %t | FileCheck %s

## If a SHT_NOBITS section is the only section of a PT_LOAD segment,
## p_offset will be set to the sh_offset field of the section. Check we align
## sh_offset to sh_addr modulo max-page-size, so that p_vaddr=p_offset (mod
## p_align).

# CHECK:      Name  Type     Address          Off     Size
# CHECK-NEXT:       NULL     0000000000000000 000000  000000
# CHECK-NEXT: .text PROGBITS 0000000000000000 000190  000000
# CHECK-NEXT: .sec1 NOBITS   0000000000000000 001000  000001
# CHECK-NEXT: .bss  NOBITS   0000000000000400 001400  000001

# CHECK:      Type Offset   VirtAddr           PhysAddr           FileSiz  MemSiz   Flg Align
# CHECK-NEXT: LOAD 0x001000 0x0000000000000000 0x0000000000000000 0x000000 0x000001 R   0x1000
# CHECK-NEXT: LOAD 0x001400 0x0000000000000400 0x0000000000000400 0x000000 0x000001 RW  0x1000

# CHECK:      00 .sec1 {{$}}
# CHECK:      01 .bss {{$}}

.bss
.p2align 10
.byte 0
