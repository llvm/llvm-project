# REQUIRES: systemz
# RUN: llvm-mc -filetype=obj -triple=s390x-unknown-linux %s -o %t1.o

## Check -pie.
# RUN: ld.lld -pie %t1.o -o %t
# RUN: llvm-readelf --file-headers --program-headers --dynamic %t | FileCheck %s

# CHECK: ELF Header:
# CHECK-NEXT:  Magic:   7f 45 4c 46 02 02 01 00 00 00 00 00 00 00 00 00
# CHECK-NEXT:  Class:                             ELF64
# CHECK-NEXT:  Data:                              2's complement, big endian
# CHECK-NEXT:  Version:                           1 (current)
# CHECK-NEXT:  OS/ABI:                            UNIX - System V
# CHECK-NEXT:  ABI Version:                       0
# CHECK-NEXT:  Type:                              DYN (Shared object file)
# CHECK-NEXT:  Machine:                           IBM S/390
# CHECK-NEXT:  Version:                           0x1

# CHECK: Program Headers:
# CHECK-NEXT:  Type           Offset   VirtAddr           PhysAddr           FileSiz  MemSiz   Flg Align
# CHECK-NEXT:  PHDR           0x000040 0x0000000000000040 0x0000000000000040 0x000188 0x000188 R   0x8
# CHECK-NEXT:  LOAD           0x000000 0x0000000000000000 0x0000000000000000 0x00020d 0x00020d R   0x1000
# CHECK-NEXT:  LOAD           0x000210 0x0000000000002210 0x0000000000002210 0x000090 0x000df0 RW  0x1000
# CHECK-NEXT:  DYNAMIC        0x000210 0x0000000000002210 0x0000000000002210 0x000090 0x000090 RW  0x8
# CHECK-NEXT:  GNU_RELRO      0x000210 0x0000000000002210 0x0000000000002210 0x000090 0x000df0 R   0x1
# CHECK-NEXT:  GNU_STACK      0x000000 0x0000000000000000 0x0000000000000000 0x000000 0x000000 RW  0x0

# CHECK: Dynamic section at offset 0x210 contains 9 entries:
# CHECK-NEXT:   Tag                Type       Name/Value
# CHECK-NEXT:   0x000000006ffffffb (FLAGS_1)  PIE

## Check -nopie
# RUN: ld.lld -no-pie %t1.o -o %t2
# RUN: llvm-readelf --file-headers %t2 | FileCheck %s --check-prefix=NOPIE
# NOPIE-NOT: Type: DYN

.globl _start
_start:
