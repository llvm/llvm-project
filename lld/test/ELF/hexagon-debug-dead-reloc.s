# REQUIRES: hexagon
## Test that R_HEX_32 relocations in .debug_* sections referencing a symbol
## discarded by --gc-sections are resolved to a tombstone value that overwrites
## (rather than bitwise-ORs into) the existing bytes at the relocation site.
##
## This is a regression test for a bug where Hexagon applied R_HEX_32 with
## or32le(), so writing a tombstone of 0 was a no-op (x | 0 == x) and the stale
## DW_AT_low_pc bytes (e.g. 0xf85c2001) of a garbage-collected inlined function
## survived into the output, causing debuggers to set breakpoints at an invalid
## address.

# RUN: llvm-mc -filetype=obj -triple=hexagon %s -o %t.o
# RUN: ld.lld --gc-sections -e live_entry %t.o -o %t
# RUN: llvm-objdump -s %t | FileCheck %s

## The relocation slots are pre-filled with 0x01 0x20 0x5c 0xf8 (little-endian
## 0xf85c2001) to prove the stale bytes are overwritten by the tombstone.
# CHECK:      Contents of section .debug_info:
# CHECK-NEXT:  0000 00000000
# CHECK-NEXT: Contents of section .debug_loc:
# CHECK-NEXT:  0000 01000000
# CHECK-NEXT: Contents of section .debug_ranges:
# CHECK-NEXT:  0000 01000000
# CHECK-NEXT: Contents of section .debug_names:
# CHECK-NEXT:  0000 ffffffff

## -z dead-reloc-in-nonalloc= can override the tombstone value, and it must also
## overwrite (not OR into) the stale bytes.
# RUN: ld.lld --gc-sections -e live_entry -z dead-reloc-in-nonalloc=.debug_info=0x42 %t.o -o %t1
# RUN: llvm-objdump -s %t1 | FileCheck %s --check-prefix=OVERRIDE

# OVERRIDE:      Contents of section .debug_info:
# OVERRIDE-NEXT:  0000 42000000

## Live (kept) code, used as the GC root via -e.
.section .text.live,"ax",@progbits
.globl live_entry
live_entry:
  jumpr r31

## Out-of-line function copy. It is not referenced by any live section, so
## --gc-sections removes it and its symbol is demoted to Undefined.
.section .text.dead,"ax",@progbits
.globl dead_func
dead_func:
  jumpr r31

## Each .debug_* slot is pre-filled with 0xf85c2001 and carries an R_HEX_32
## relocation against the discarded symbol.
##
## .debug_info: tombstone 0.
.section .debug_info,"",@progbits
di:
  .byte 0x01, 0x20, 0x5c, 0xf8
  .reloc di, R_HEX_32, dead_func

## .debug_loc: tombstone 1 (0/-1 are reserved base-address selection entries).
.section .debug_loc,"",@progbits
dl:
  .byte 0x01, 0x20, 0x5c, 0xf8
  .reloc dl, R_HEX_32, dead_func

## .debug_ranges: tombstone 1.
.section .debug_ranges,"",@progbits
dr:
  .byte 0x01, 0x20, 0x5c, 0xf8
  .reloc dr, R_HEX_32, dead_func

## .debug_names: tombstone UINT32_MAX.
.section .debug_names,"",@progbits
dn:
  .byte 0x01, 0x20, 0x5c, 0xf8
  .reloc dn, R_HEX_32, dead_func
