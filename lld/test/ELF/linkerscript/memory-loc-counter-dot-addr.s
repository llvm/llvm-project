# REQUIRES: x86
# RUN: rm -rf %t && split-file %s %t && cd %t
# RUN: llvm-mc -filetype=obj -triple=x86_64 asm.s -o t.o
# RUN: ld.lld -T lds t.o -o t.out
# RUN: llvm-readelf -S t.out | FileCheck %s

## Test that when a section uses '.' as its address expression inside a MEMORY
## region, the global location counter is used, not the memory region position.
## GNU ld manual: explicit address takes precedence over memory region position.
## s01 placed at FLASH origin 0x1000
# CHECK: s01 PROGBITS 0000000000001000
## s02 uses '.' as explicit address — global dot after ALIGN(4) = 0x1004
# CHECK: s02 PROGBITS 0000000000001004
## s03 has no explicit address — follows memRegion->curPos after s02
## second ALIGN(4) does not affect s03 as it has no explicit address
# CHECK: s03 PROGBITS 0000000000001006

#--- asm.s
.section s01, "a", @progbits
.byte 1

.section s02, "a", @progbits
.short 1

.section s03, "a", @progbits
.short 1

#--- lds
MEMORY
{
  FLASH (rx) : ORIGIN = 0x1000, LENGTH = 0x100
}

SECTIONS
{
  s01 : { *(s01) } > FLASH
  . = ALIGN(4);
  s02 . : { *(s02) } > FLASH
  . = ALIGN(4);
  s03 : { *(s03) } > FLASH
}
