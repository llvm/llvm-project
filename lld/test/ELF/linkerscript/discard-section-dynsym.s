# REQUIRES: aarch64

## We allow discarding .dynsym, check we don't crash.
# RUN: rm -rf %t && split-file %s %t && cd %t
# RUN: llvm-mc -filetype=obj -triple=aarch64 a.s -o a.o
# RUN: llvm-mc -filetype=obj -triple=aarch64 c.s -o c.o
# RUN: ld.lld -shared --version-script=c.ver c.o -o c.so

# RUN: echo 'SECTIONS { /DISCARD/ : { *(.dynsym) } }' > 1.lds
# RUN: ld.lld -shared -T 1.lds a.o c.so -o out1.so
# RUN: llvm-readelf -Sr out1.so | FileCheck %s --check-prefixes=CHECK,CHECK1

# RUN: echo 'SECTIONS { /DISCARD/ : { *(.dynsym .dynstr) } }' > 2.lds
# RUN: ld.lld -shared -T 2.lds a.o c.so -o out2.so
# RUN: llvm-readelf -Sr out2.so | FileCheck %s --check-prefixes=CHECK,CHECK2

# CHECK:       [Nr] Name              Type            Address          Off    Size   ES Flg Lk Inf Al
# CHECK-NEXT:  [ 0]                   NULL            0000000000000000 000000 000000 00      0   0  0
# CHECK-NEXT:  [ 1] .gnu.version      VERSYM          0000000000000000 {{.*}} 000006 02   A  0   0  2
# CHECK1-NEXT: [ 2] .gnu.version_r    VERNEED         0000000000000008 {{.*}} 000020 00   A  5   1  4
# CHECK2-NEXT: [ 2] .gnu.version_r    VERNEED         0000000000000008 {{.*}} 000020 00   A  0   1  4
# CHECK1:      [ 5] .dynstr           STRTAB

# CHECK:      contains 2 entries:
# CHECK:      R_AARCH64_RELATIVE  [[#]]
# CHECK-NEXT: R_AARCH64_GLOB_DAT  0{{$}}

#--- a.s
  adrp x9, :got:var
  ldr  x9, [x9, :got_lo12:var]
  bl __libc_start_main

.data
.align 8
foo:
.quad foo

#--- c.s
.globl __libc_start_main
__libc_start_main:

#--- c.ver
GLIBC_2.34 { __libc_start_main; };
