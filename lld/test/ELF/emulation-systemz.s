# REQUIRES: systemz
# RUN: llvm-mc -filetype=obj -triple=s390x-unknown-linux %s -o %t.o
# RUN: ld.lld -m elf64_s390 %t.o -o %t1
# RUN: llvm-readelf --file-header %t1 | FileCheck %s
# RUN: ld.lld %t.o -o %t2
# RUN: llvm-readelf --file-header %t2 | FileCheck %s
# RUN: echo 'OUTPUT_FORMAT(elf64-s390)' > %t.script
# RUN: ld.lld %t.script %t.o -o %t3
# RUN: llvm-readelf --file-header %t3 | FileCheck %s

# CHECK:       ELF Header:
# CHECK-NEXT:  Magic:   7f 45 4c 46 02 02 01 00 00 00 00 00 00 00 00 00
# CHECK-NEXT:  Class:                             ELF64
# CHECK-NEXT:  Data:                              2's complement, big endian
# CHECK-NEXT:  Version:                           1 (current)
# CHECK-NEXT:  OS/ABI:                            UNIX - System V
# CHECK-NEXT:  ABI Version:                       0
# CHECK-NEXT:  Type:                              EXEC (Executable file)
# CHECK-NEXT:  Machine:                           IBM S/390
# CHECK-NEXT:  Version:                           0x1
# CHECK-NEXT:  Entry point address:
# CHECK-NEXT:  Start of program headers:          64 (bytes into file)
# CHECK-NEXT:  Start of section headers:
# CHECK-NEXT:  Flags:                             0x0
# CHECK-NEXT:  Size of this header:               64 (bytes)
# CHECK-NEXT:  Size of program headers:           56 (bytes)

.globl _start
_start:
