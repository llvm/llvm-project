# REQUIRES: x86, zlib

# RUN: llvm-mc -compress-debug-sections=zlib -filetype=obj -triple=x86_64-unknown-linux %s -o %t.o
# RUN: llvm-readobj --sections %t.o | FileCheck %s --check-prefix=INPUT
# RUN: ld.lld %t.o -o %t
# RUN: llvm-readobj --sections --section-data %t | FileCheck %s

## -r and --emit-relocs call copyRelocations, which decompresses the section
## before it is written.
# RUN: ld.lld --emit-relocs %t.o -o %t.emit-relocs
# RUN: llvm-readobj --sections %t.emit-relocs | FileCheck %s --check-prefix=DECOMPRESSED
# RUN: ld.lld -r %t.o -o %t.r
# RUN: llvm-readobj --sections %t.r | FileCheck %s --check-prefix=DECOMPRESSED

## `.debug_gdb_scripts` is emitted with SHF_ALLOC, and MC compresses every
## `.debug_*` section, so lld must accept an SHF_ALLOC | SHF_COMPRESSED input.

# INPUT:        Name: .debug_gdb_scripts
# INPUT-NEXT:   Type: SHT_PROGBITS
# INPUT-NEXT:   Flags [
# INPUT-NEXT:     SHF_ALLOC (0x2)
# INPUT-NEXT:     SHF_COMPRESSED (0x800)
# INPUT-NEXT:   ]

# DECOMPRESSED:        Name: .debug_gdb_scripts
# DECOMPRESSED-NEXT:   Type: SHT_PROGBITS
# DECOMPRESSED-NEXT:   Flags [
# DECOMPRESSED-NEXT:     SHF_ALLOC (0x2)
# DECOMPRESSED-NEXT:   ]

# CHECK:        Name: .debug_gdb_scripts
# CHECK-NEXT:   Type: SHT_PROGBITS
# CHECK-NEXT:   Flags [
# CHECK-NEXT:     SHF_ALLOC (0x2)
# CHECK-NEXT:   ]
# CHECK-NEXT:   Address: 0x200120
# CHECK-NEXT:   Offset: 0x120
# CHECK-NEXT:   Size: 74
# CHECK-NEXT:   Link: 0
# CHECK-NEXT:   Info: 0
# CHECK-NEXT:   AddressAlignment: 8
# CHECK-NEXT:   EntrySize: 0
# CHECK-NEXT:   SectionData (
# CHECK-NEXT:     0000: {{[0-9A-F]+ [0-9A-F]+}} 76616C75 652D7072
# CHECK-NEXT:     0010: 696E7465 72207061 6464696E 67007661  |inter padding.va|
# CHECK-NEXT:     0020: 6C75652D 7072696E 74657220 70616464  |lue-printer padd|
# CHECK-NEXT:     0030: 696E6700 76616C75 652D7072 696E7465  |ing.value-printe|
# CHECK-NEXT:     0040: 72207061 6464696E 6700               |r padding.|
# CHECK-NEXT:   )

.section .debug_gdb_scripts,"a",@progbits
.p2align 3
.quad target
.rept 3
.asciz "value-printer padding"
.endr
.text
target:
  ret
