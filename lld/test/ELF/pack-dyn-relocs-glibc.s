# REQUIRES: x86
## -z pack-relative-relocs is a variant of --pack-dyn-relocs=relr: add
## GLIBC_ABI_DT_RELR verneed if there is a verneed named "GLIBC_2.*".

# RUN: rm -rf %t && split-file %s %t
# RUN: llvm-mc -filetype=obj -triple=x86_64 %t/a.s -o %t/a.o
# RUN: llvm-mc -filetype=obj -triple=x86_64 %t/libc.s -o %t/libc.o
# RUN: ld.lld -shared --soname=libc.so.6 --version-script=%t/glibc.ver %t/libc.o -o %t/libc.so.6

# RUN: ld.lld -pie %t/a.o %t/libc.so.6 -z pack-relative-relocs -o %t/glibc 2>&1 | count 0
# RUN: llvm-readelf -r -V %t/glibc | FileCheck %s --check-prefix=GLIBC
## Arbitrarily let -z pack-relative-relocs win.
# RUN: ld.lld -pie %t/a.o %t/libc.so.6 -z pack-relative-relocs --pack-dyn-relocs=relr -o %t/glibc2
# RUN: cmp %t/glibc %t/glibc2

# GLIBC:      Relocation section '.relr.dyn' at offset {{.*}} contains 1 entries:
# GLIBC:      Version needs section '.gnu.version_r' contains 1 entries:
# GLIBC-NEXT:  Addr: {{.*}}
# GLIBC-NEXT:   0x0000: Version: 1  File: libc.so.6  Cnt: 2
# GLIBC-NEXT:   0x0010:   Name: GLIBC_2.33  Flags: none  Version: 2
# GLIBC-NEXT:   0x0020:   Name: GLIBC_ABI_DT_RELR  Flags: none  Version: 3
# GLIBC-EMPTY:

# RUN: ld.lld -pie %t/a.o %t/libc.so.6 -z pack-relative-relocs -z nopack-relative-relocs -o %t/notrelr 2>&1 | count 0
# RUN: llvm-readelf -r -V %t/notrelr | FileCheck %s --check-prefix=REGULAR

# REGULAR-NOT: Relocation section '.relr.dyn'
# REGULAR-NOT: Name: GLIBC_ABI_DT_RELR

## soname is not "libc.so.*". Don't synthesize GLIBC_ABI_DT_RELR. In glibc, ld.so
## doesn't define GLIBC_ABI_DT_RELR. libc.so itself should not reference GLIBC_ABI_DT_RELR.
# RUN: ld.lld -shared --soname=ld-linux-x86-64.so.2 --version-script=%t/glibc.ver %t/libc.o -o %t/ld.so
# RUN: ld.lld -pie %t/a.o %t/ld.so -z pack-relative-relocs -o %t/other 2>&1 | count 0
# RUN: llvm-readelf -r -V %t/other | FileCheck %s --check-prefix=NOTLIBC

# NOTLIBC:     Relocation section '.relr.dyn' at offset {{.*}} contains 1 entries:
# NOTLIBC-NOT: Name: GLIBC_ABI_DT_RELR

## There is no GLIBC_2.* verneed. Don't add GLIBC_ABI_DT_RELR verneed.
# RUN: ld.lld -shared --soname=libc.so.6 --version-script=%t/other.ver %t/libc.o -o %t/libc.so.6
# RUN: ld.lld -pie %t/a.o %t/libc.so.6 -z pack-relative-relocs -o %t/other
# RUN: llvm-readelf -r -V %t/other | FileCheck %s --check-prefix=NOTLIBC

#--- a.s
.globl _start
_start:
  call stat

.data
.balign 8
.dc.a .data

#--- libc.s
.weak stat
stat:

#--- glibc.ver
GLIBC_2.33 {
  stat;
};

#--- other.ver
GLIBC_3 {
  stat;
};
