# REQUIRES: hexagon
# RUN: llvm-mc -filetype=obj -triple=hexagon %s -o %t.o
# RUN: ld.lld %t.o -o %t
# RUN: llvm-readelf --file-headers %t | FileCheck --check-prefix=CHECK %s
# RUN: ld.lld -m hexagonelf %t.o -o %t
# RUN: llvm-readelf --file-headers %t | FileCheck --check-prefix=CHECK %s

# RUN: echo 'OUTPUT_FORMAT(elf32-littlehexagon)' > %t.script
# RUN: ld.lld %t.script %t.o -o %t
# RUN: llvm-readelf --file-headers %t | FileCheck --check-prefix=CHECK %s

# RUN: echo 'OUTPUT_FORMAT(elf32-hexagon)' > %t.script
# RUN: ld.lld %t.script %t.o -o %t
# RUN: llvm-readelf --file-headers %t | FileCheck --check-prefix=CHECK %s

# CHECK:       ELF Header:
# CHECK-NEXT:    Magic:   7f 45 4c 46 01 01 01 00 00 00 00 00 00 00 00 00
# CHECK-NEXT:    Class:                             ELF32
# CHECK-NEXT:    Data:                              2's complement, little endian
# CHECK-NEXT:    Version:                           1 (current)
# CHECK-NEXT:    OS/ABI:                            UNIX - System V
# CHECK-NEXT:    ABI Version:                       0
# CHECK-NEXT:    Type:                              EXEC (Executable file)
# CHECK-NEXT:    Machine:                           Qualcomm Hexagon
# CHECK-NEXT:    Version:                           0x1
# CHECK-NEXT:    Entry point address:               0x200B4
# CHECK-NEXT:    Start of program headers:          52 (bytes into file)
# CHECK-NEXT:    Start of section headers:
# CHECK-NEXT:    Flags:                             0x60
# CHECK-NEXT:    Size of this header:               52 (bytes)
# CHECK-NEXT:    Size of program headers:           32 (bytes)

.globl _start
_start:
