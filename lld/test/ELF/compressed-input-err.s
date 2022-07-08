# REQUIRES: zlib
# RUN: yaml2obj --docnum=1 %s -o %t1.o
# RUN: not ld.lld %t1.o -o /dev/null 2>&1 | FileCheck %s --check-prefix=TOO-SHORT
# TOO-SHORT: error: {{.*}}.o:(.debug_info): corrupted compressed section

# RUN: yaml2obj --docnum=2 %s -o %t2.o
# RUN: not ld.lld %t2.o -o /dev/null 2>&1 | FileCheck %s --check-prefix=UNKNOWN
# UNKNOWN: error: {{.*}}.o:(.debug_info): unsupported compression type (3)

# RUN: yaml2obj --docnum=3 %s -o %t3.o
# RUN: not ld.lld %t3.o -o /dev/null -shared 2>&1 | FileCheck %s

## Check we are able to report zlib uncompress errors.
# CHECK: error: {{.*}}.o:(.debug_info): uncompress failed: zlib error: Z_DATA_ERROR

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
    Content:      "0100000000000000040000000000000001000000000000"

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
    Content:      "030000000000000000000000000000000100000000000000789c030000000001"

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
    Content:      "010000000000000004000000000000000100000000000000ffff"
