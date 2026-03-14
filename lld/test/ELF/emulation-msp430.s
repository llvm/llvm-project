# REQUIRES: msp430
# RUN: llvm-mc -filetype=obj -triple=msp430 %s -o %t.o
# RUN: ld.lld %t.o -o %t1
# RUN: llvm-readobj --file-headers %t1 | FileCheck %s
# RUN: ld.lld -m msp430elf %t.o -o %t2
# RUN: cmp %t1 %t2
# RUN: echo 'OUTPUT_FORMAT(elf32-msp430)' > %t.lds
# RUN: ld.lld -T %t.lds %t.o -o %t3
# RUN: llvm-readobj --file-headers %t3 | FileCheck %s

# CHECK:      ElfHeader {
# CHECK-NEXT:   Ident {
# CHECK-NEXT:     Magic: (7F 45 4C 46)
# CHECK-NEXT:     Class: 32-bit (0x1)
# CHECK-NEXT:     DataEncoding: LittleEndian (0x1)
# CHECK-NEXT:     FileVersion: 1
# CHECK-NEXT:     OS/ABI: Standalone (0xFF)
# CHECK-NEXT:     ABIVersion: 0
# CHECK-NEXT:     Unused: (00 00 00 00 00 00 00)
# CHECK-NEXT:   }
# CHECK-NEXT:   Type: Executable (0x2)
# CHECK-NEXT:   Machine: EM_MSP430 (0x69)
# CHECK-NEXT:   Version: 1

.globl _start
_start:
