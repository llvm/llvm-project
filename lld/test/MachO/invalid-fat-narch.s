# REQUIRES: x86
# RUN: yaml2obj %s -o %t.o
# RUN: not lld -flavor darwinnew -arch x86_64 -o /dev/null %t.o 2>&1 | \
# RUN:    FileCheck %s -DFILE=%t.o
# CHECK: error: [[FILE]]: fat_arch struct extends beyond end of file

!fat-mach-o
FatHeader:
  magic:           0xCAFEBABE
  nfat_arch:       2
FatArchs:
Slices:
