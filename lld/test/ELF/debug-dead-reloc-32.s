# REQUIRES: x86
## Test we resolve symbolic relocations in .debug_* sections to a tombstone
## value if the referenced symbol is discarded (--gc-sections, non-prevailing
## section group, SHF_EXCLUDE, /DISCARD/, etc).

# RUN: llvm-mc -filetype=obj -triple=i386 %s -o %t.o
# RUN: ld.lld %t.o -o %t
# RUN: llvm-objdump -s %t | FileCheck %s

# CHECK:      Contents of section .debug_loc:
# CHECK-NEXT:  0000 feffffff
# CHECK-NEXT: Contents of section .debug_ranges:
# CHECK-NEXT:  0000 feffffff
# CHECK-NEXT: Contents of section .debug_addr:
# CHECK-NEXT:  0000 ffffffff

.section .text.1,"axe"
  .byte 0

## Resolved to UINT32_C(-2), with the addend ignored.
## UINT32_C(-1) is a reserved value (base address selection entry) which can't be used.
.section .debug_loc
  .long .text.1+8
.section .debug_ranges
  .long .text.1+16

## Resolved to UINT32_C(-1), with the addend ignored.
.section .debug_addr
  .long .text.1+8
