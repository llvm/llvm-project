# REQUIRES: x86
## Test we resolve symbolic relocations in .debug_* sections to a tombstone
## value if the referenced symbol is discarded (--gc-sections, non-prevailing
## section group, SHF_EXCLUDE, /DISCARD/, etc).

# RUN: llvm-mc -filetype=obj -triple=i386 %s -o %t.o
# RUN: ld.lld %t.o -o %t
# RUN: llvm-objdump -s %t | FileCheck %s

# CHECK:      Contents of section .debug_loc:
# CHECK-NEXT:  0000 01000000
# CHECK-NEXT: Contents of section .debug_ranges:
# CHECK-NEXT:  0000 01000000
# CHECK-NEXT: Contents of section .debug_addr:
# CHECK-NEXT:  0000 00000000
# CHECK-NEXT: Contents of section .debug_names:
# CHECK-NEXT:  0000 ffffffff

## -z dead-reloc-in-nonalloc= can override the tombstone value.
# RUN: ld.lld -z dead-reloc-in-nonalloc=.debug_loc=42 -z dead-reloc-in-nonalloc=.debug_addr=0xfffffffffffffffe %t.o -o %t1
# RUN: llvm-objdump -s %t1 | FileCheck %s --check-prefix=OVERRIDE

# OVERRIDE:      Contents of section .debug_loc:
# OVERRIDE-NEXT:  0000 2a000000                             *...
# OVERRIDE-NEXT: Contents of section .debug_ranges:
# OVERRIDE-NEXT:  0000 01000000                             ....
# OVERRIDE-NEXT: Contents of section .debug_addr:
# OVERRIDE-NEXT:  0000 feffffff                             ....

.section .text.1,"axe"
  .byte 0

## Resolved to UINT32_C(-2), with the addend ignored.
## UINT32_C(-1) is a reserved value (base address selection entry) which can't be used.
.section .debug_loc
  .long .text.1+8
.section .debug_ranges
  .long .text.1+16

## Resolved to UINT32_C(0), with the addend ignored.
.section .debug_addr
  .long .text.1+8

.section  .debug_info,"eG",@progbits,5657452045627120676,comdat
.Ltu_begin0:

.section .debug_names
## .debug_names may reference a local type unit defined in a COMDAT .debug_info
## section (-g -gpubnames -fdebug-types-section). If the referenced section is
## non-prevailing, resolve to UINT32_MAX.
.long .Ltu_begin0
