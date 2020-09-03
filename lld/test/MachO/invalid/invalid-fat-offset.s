# REQUIRES: x86
# RUN: yaml2obj %s -o %t.o
# RUN: not lld -flavor darwinnew -o /dev/null %t.o 2>&1 | \
# RUN:    FileCheck %s -DFILE=%t.o
# CHECK: error: [[FILE]]: slice extends beyond end of file

!fat-mach-o
FatHeader:
  magic:           0xCAFEBABE
  nfat_arch:       2
FatArchs:
  - cputype:         0x01000007
    cpusubtype:      0x00000003
    offset:          0x0000000000001000
    size:            0
    align:           12
  - cputype:         0x00000007
    cpusubtype:      0x00000003
    offset:          0x000000000000B000
    size:            0
    align:           12
Slices:
