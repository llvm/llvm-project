# REQUIRES: zlib, ppc, x86

# RUN: llvm-mc -compress-debug-sections=zlib -filetype=obj -triple=x86_64-unknown-linux %s -o %t
# RUN: llvm-mc -compress-debug-sections=zlib -filetype=obj -triple=powerpc64-unknown-unknown %s -o %t-be
# RUN: llvm-readobj --sections %t | FileCheck -check-prefix=ZLIB %s
# RUN: llvm-readobj --sections %t-be | FileCheck -check-prefix=ZLIB %s
# ZLIB:      Section {
# ZLIB:        Index: 2
# ZLIB:        Name: .debug_str
# ZLIB-NEXT:   Type: SHT_PROGBITS
# ZLIB-NEXT:   Flags [
# ZLIB-NEXT:     SHF_COMPRESSED (0x800)
# ZLIB-NEXT:     SHF_MERGE (0x10)
# ZLIB-NEXT:     SHF_STRINGS (0x20)
# ZLIB-NEXT:   ]
# ZLIB-NEXT:   Address:
# ZLIB-NEXT:   Offset:
# ZLIB-NEXT:   Size:
# ZLIB-NEXT:   Link:
# ZLIB-NEXT:   Info:
# ZLIB-NEXT:   AddressAlignment: 8
# ZLIB-NEXT:   EntrySize: 1
# ZLIB-NEXT: }

# RUN: ld.lld --hash-style=sysv %t -o %t.so -shared
# RUN: llvm-readobj --sections --section-data %t.so | FileCheck -check-prefix=DATA %s
# RUN: ld.lld --hash-style=sysv %t-be -o %t-be.so -shared
# RUN: llvm-readobj --sections --section-data %t-be.so | FileCheck -check-prefix=DATA %s

# DATA:      Section {
# DATA:        Index: 6
# DATA:        Name: .debug_str
# DATA-NEXT:   Type: SHT_PROGBITS
# DATA-NEXT:   Flags [
# DATA-NEXT:     SHF_MERGE (0x10)
# DATA-NEXT:     SHF_STRINGS (0x20)
# DATA-NEXT:   ]
# DATA-NEXT:   Address: 0x0
# DATA-NEXT:   Offset:
# DATA-NEXT:   Size: 69
# DATA-NEXT:   Link: 0
# DATA-NEXT:   Info: 0
# DATA-NEXT:   AddressAlignment: 1
# DATA-NEXT:   EntrySize: 1
# DATA-NEXT:   SectionData (
# DATA-NEXT:     0000: 73686F72 7420756E 7369676E 65642069  |short unsigned i|
# DATA-NEXT:     0010: 6E740075 6E736967 6E656420 63686172  |nt.unsigned char|
# DATA-NEXT:     0020: 00636861 72006C6F 6E672075 6E736967  |.char.long unsig|
# DATA-NEXT:     0030: 6E656420 696E7400 756E7369 676E6564  |ned int.unsigned|
# DATA-NEXT:     0040: 20696E74 00                          | int.|
# DATA-NEXT:   )
# DATA-NEXT: }

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
