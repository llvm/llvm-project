# UNSUPPORTED: zstd
# RUN: yaml2obj %s -o %t.o
# RUN: not ld.lld %t.o -o /dev/null 2>&1 | FileCheck %s

# CHECK: error: {{.*}}.o:(.debug_info) is compressed with ELFCOMPRESS_ZSTD, but lld is not built with zstd support

--- !ELF
FileHeader:
  Class:   ELFCLASS64
  Data:    ELFDATA2LSB
  Type:    ET_REL
  Machine: EM_X86_64
Sections:
  - Type:         SHT_PROGBITS
    Name:         .debug_info
    Flags:        [ SHF_COMPRESSED ]
    AddressAlign: 8
    Content:      "020000000000000000000000000000000100000000000000789c030000000001"
