# REQUIRES: ppc
# RUN: llvm-mc -filetype=obj -triple=powerpc64le %s -o %tle.o
# RUN: ld.lld --hash-style=sysv -discard-all -shared %tle.o -o %tle.so
# RUN: llvm-readelf -hSl %tle.so | FileCheck --check-prefixes=CHECK,LE %s

# RUN: llvm-mc -filetype=obj -triple=powerpc64 %s -o %tbe.o
# RUN: ld.lld --hash-style=sysv -discard-all -shared %tbe.o -o %tbe.so
# RUN: llvm-readelf -hSl %tbe.so | FileCheck --check-prefixes=CHECK,BE %s

# CHECK:        Class:                             ELF64
# LE-NEXT:      Data:                              2's complement, little endian
# BE-NEXT:      Data:                              2's complement, big endian
# CHECK-NEXT:   Version:                           1 (current)
# CHECK-NEXT:   OS/ABI:                            UNIX - System V
# CHECK-NEXT:   ABI Version:                       0
# CHECK-NEXT:   Type:                              DYN (Shared object file)
# CHECK-NEXT:   Machine:                           PowerPC64

# CHECK:      Name              Type            Address          Off    Size   ES Flg Lk Inf Al
# CHECK:      .branch_lt        NOBITS          {{.*}}                  000000 00  WA  0   0  8

.abiversion 2
# Exits with return code 55 on linux.
.text
  li 0,1
  li 3,55
  sc


