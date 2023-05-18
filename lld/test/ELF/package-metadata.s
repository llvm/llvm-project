# REQUIRES: x86

# RUN: llvm-mc -filetype=obj -triple=x86_64 %s -o %t.o

# RUN: ld.lld %t.o -o %t --package-metadata='{}'
# RUN: llvm-readelf -n %t | FileCheck %s --check-prefixes=NOTE,FIRST

# RUN: ld.lld %t.o -o %t --package-metadata='{"abc":123}'
# RUN: llvm-readelf -n %t | FileCheck %s --check-prefixes=NOTE,SECOND

# NOTE: .note.package
# NOTE-NEXT: Owner
# NOTE-NEXT: FDO 0x{{.*}} Unknown note type: (0xcafe1a7e)
# FIRST-NEXT: description data: 7b 7d 00
# SECOND-NEXT: description data: 7b 22 61 62 63 22 3a 31 32 33 7d 00

.globl _start
_start:
  ret
