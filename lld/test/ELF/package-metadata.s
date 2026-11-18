# REQUIRES: x86
# RUN: rm -rf %t && split-file %s %t && cd %t
# RUN: llvm-mc -filetype=obj -triple=x86_64 a.s -o a.o

# RUN: ld.lld a.o --package-metadata='{}'
# RUN: llvm-readelf -n a.out | FileCheck %s --check-prefixes=NOTE,FIRST

# RUN: ld.lld a.o --package-metadata='{"abc":123}'
# RUN: llvm-readelf -n a.out | FileCheck %s --check-prefixes=NOTE,SECOND

# RUN: ld.lld a.o --package-metadata='%7b%22abc%22:123%7D'
# RUN: llvm-readelf -n a.out | FileCheck %s --check-prefixes=NOTE,SECOND

# NOTE: .note.package
# NOTE-NEXT: Owner
# NOTE-NEXT: FDO 0x{{.*}} FDO_PACKAGING_METADATA
# FIRST-NEXT: Packaging Metadata: {}
# SECOND-NEXT: Packaging Metadata: {"abc":123}

# RUN: not ld.lld a.o --package-metadata='%7b%' 2>&1 | FileCheck %s --check-prefix=ERR
# RUN: not ld.lld a.o --package-metadata='%7b%7' 2>&1 | FileCheck %s --check-prefix=ERR
# RUN: not ld.lld a.o --package-metadata='%7b%7g' 2>&1 | FileCheck %s --check-prefix=ERR

# ERR: error: --package-metadata=: invalid % escape at byte 3; supports only %[0-9a-fA-F][0-9a-fA-F]

#--- a.s
.globl _start
_start:
  ret
