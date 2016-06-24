# REQUIRES: zlib

# RUN: llvm-mc -compress-debug-sections=zlib -filetype=obj -triple=x86_64-unknown-linux %s -o %t
# RUN: llvm-readobj -sections %t | FileCheck -check-prefix=COMPRESSED %s

# COMPRESSED:      Section {
# COMPRESSED:        Index: 2
# COMPRESSED:        Name: .debug_str
# COMPRESSED-NEXT:   Type: SHT_PROGBITS
# COMPRESSED-NEXT:   Flags [
# COMPRESSED-NEXT:     SHF_COMPRESSED (0x800)
# COMPRESSED-NEXT:     SHF_MERGE (0x10)
# COMPRESSED-NEXT:     SHF_STRINGS (0x20)
# COMPRESSED-NEXT:   ]
# COMPRESSED-NEXT:   Address:
# COMPRESSED-NEXT:   Offset:
# COMPRESSED-NEXT:   Size: 66
# COMPRESSED-NEXT:   Link:
# COMPRESSED-NEXT:   Info:
# COMPRESSED-NEXT:   AddressAlignment: 1
# COMPRESSED-NEXT:   EntrySize: 1
# COMPRESSED-NEXT: }

# RUN: ld.lld %t -o %t.so -shared
# RUN: llvm-readobj -sections %t.so | FileCheck -check-prefix=UNCOMPRESSED %s

## Check that section is decompressed and compression flag is removed.
# UNCOMPRESSED:      Section {
# UNCOMPRESSED:        Index: 6
# UNCOMPRESSED:        Name: .debug_str
# UNCOMPRESSED-NEXT:   Type: SHT_PROGBITS
# UNCOMPRESSED-NEXT:   Flags [
# UNCOMPRESSED-NEXT:     SHF_MERGE (0x10)
# UNCOMPRESSED-NEXT:     SHF_STRINGS (0x20)
# UNCOMPRESSED-NEXT:   ]
# UNCOMPRESSED-NEXT:   Address: 0x0
# UNCOMPRESSED-NEXT:   Offset: 0x1060
# UNCOMPRESSED-NEXT:   Size: 69
# UNCOMPRESSED-NEXT:   Link: 0
# UNCOMPRESSED-NEXT:   Info: 0
# UNCOMPRESSED-NEXT:   AddressAlignment: 1
# UNCOMPRESSED-NEXT:   EntrySize: 1
# UNCOMPRESSED-NEXT: }

.section .debug_str,"MS",@progbits,1
.LASF2:
 .string "short unsigned int"
.LASF3:
 .string "unsigned int"
.LASF0:
 .string "long unsigned int"
.LASF8:
 .string "char"
.LASF1:
 .string "unsigned char"
