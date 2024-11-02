# REQUIRES: zlib
# RUN: yaml2obj %s -o %t.o
# RUN: not ld.lld %t.o -o /dev/null -shared 2>&1 | FileCheck %s

## Check we are able to report zlib uncompress errors.
# CHECK: error: {{.*}}.o:(.debug_info): uncompress failed: zlib error: Z_DATA_ERROR

!ELF
FileHeader:
  Class:   ELFCLASS64
  Data:    ELFDATA2LSB
  Type:    ET_REL
  Machine: EM_X86_64
Sections:
  - Type:         SHT_PROGBITS
    Name:         .debug_info
    Flags:        [ SHF_COMPRESSED ]
    AddressAlign: 0x04
    Content:      "010000000000000004000000000000000100000000000000ffff"
