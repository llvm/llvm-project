# REQUIRES: x86
# RUN: llvm-mc -filetype=obj -triple=x86_64-unknown-linux %s -o %t1.o

## Default is no PIE.
# RUN: ld.lld %t1.o -o %t
# RUN: llvm-readobj -file-headers -sections -program-headers -symbols -r %t \
# RUN:   | FileCheck %s --check-prefix=NOPIE

## Check -pie.
# RUN: ld.lld -pie %t1.o -o %t
# RUN: llvm-readobj -file-headers -sections -program-headers -symbols -r %t | FileCheck %s

## Test --pic-executable alias
# RUN: ld.lld --pic-executable %t1.o -o %t
# RUN: llvm-readobj -file-headers -sections -program-headers -symbols -r %t | FileCheck %s

# CHECK:      ElfHeader {
# CHECK-NEXT:  Ident {
# CHECK-NEXT:    Magic: (7F 45 4C 46)
# CHECK-NEXT:    Class: 64-bit
# CHECK-NEXT:    DataEncoding: LittleEndian
# CHECK-NEXT:    FileVersion: 1
# CHECK-NEXT:    OS/ABI: SystemV
# CHECK-NEXT:    ABIVersion: 0
# CHECK-NEXT:    Unused: (00 00 00 00 00 00 00)
# CHECK-NEXT:  }
# CHECK-NEXT:  Type: SharedObject
# CHECK-NEXT:  Machine: EM_X86_64
# CHECK-NEXT:  Version: 1
# CHECK-NEXT:  Entry: 0x1000
# CHECK-NEXT:  ProgramHeaderOffset: 0x40
# CHECK-NEXT:  SectionHeaderOffset: 0x1120
# CHECK-NEXT:  Flags [
# CHECK-NEXT:  ]
# CHECK-NEXT:  HeaderSize: 64
# CHECK-NEXT:  ProgramHeaderEntrySize: 56
# CHECK-NEXT:  ProgramHeaderCount: 7
# CHECK-NEXT:  SectionHeaderEntrySize: 64
# CHECK-NEXT:  SectionHeaderCount: 10
# CHECK-NEXT:  StringTableSectionIndex: 8
# CHECK-NEXT: }

# CHECK:      ProgramHeaders [
# CHECK-NEXT:  ProgramHeader {
# CHECK-NEXT:    Type: PT_PHDR
# CHECK-NEXT:    Offset: 0x40
# CHECK-NEXT:    VirtualAddress: 0x40
# CHECK-NEXT:    PhysicalAddress: 0x40
# CHECK-NEXT:    FileSize: 392
# CHECK-NEXT:    MemSize: 392
# CHECK-NEXT:    Flags [
# CHECK-NEXT:      PF_R
# CHECK-NEXT:    ]
# CHECK-NEXT:    Alignment: 8
# CHECK-NEXT:  }
# CHECK-NEXT:  ProgramHeader {
# CHECK-NEXT:    Type: PT_LOAD
# CHECK-NEXT:    Offset: 0x0
# CHECK-NEXT:    VirtualAddress: 0x0
# CHECK-NEXT:    PhysicalAddress: 0x0
# CHECK-NEXT:    FileSize: 497
# CHECK-NEXT:    MemSize: 497
# CHECK-NEXT:    Flags [
# CHECK-NEXT:      PF_R
# CHECK-NEXT:    ]
# CHECK-NEXT:    Alignment: 4096
# CHECK-NEXT:  }
# CHECK-NEXT:  ProgramHeader {
# CHECK-NEXT:    Type: PT_LOAD
# CHECK-NEXT:    Offset: 0x1000
# CHECK-NEXT:    VirtualAddress: 0x1000
# CHECK-NEXT:    PhysicalAddress: 0x1000
# CHECK-NEXT:    FileSize: 0
# CHECK-NEXT:    MemSize: 0
# CHECK-NEXT:    Flags [
# CHECK-NEXT:      PF_R
# CHECK-NEXT:      PF_X
# CHECK-NEXT:    ]
# CHECK-NEXT:    Alignment: 4096
# CHECK-NEXT:  }
# CHECK-NEXT:  ProgramHeader {
# CHECK-NEXT:    Type: PT_LOAD
# CHECK-NEXT:    Offset: 0x1000
# CHECK-NEXT:    VirtualAddress: 0x1000
# CHECK-NEXT:    PhysicalAddress: 0x1000
# CHECK-NEXT:    FileSize: 112
# CHECK-NEXT:    MemSize: 112
# CHECK-NEXT:    Flags [
# CHECK-NEXT:      PF_R
# CHECK-NEXT:      PF_W
# CHECK-NEXT:    ]
# CHECK-NEXT:    Alignment: 4096
# CHECK-NEXT:  }
# CHECK-NEXT:  ProgramHeader {
# CHECK-NEXT:    Type: PT_DYNAMIC
# CHECK-NEXT:    Offset: 0x1000
# CHECK-NEXT:    VirtualAddress: 0x1000
# CHECK-NEXT:    PhysicalAddress: 0x1000
# CHECK-NEXT:    FileSize: 112
# CHECK-NEXT:    MemSize: 112
# CHECK-NEXT:    Flags [
# CHECK-NEXT:      PF_R
# CHECK-NEXT:      PF_W
# CHECK-NEXT:    ]
# CHECK-NEXT:    Alignment: 8
# CHECK-NEXT:  }

## Check -nopie
# RUN: ld.lld -nopie %t1.o -o %t2
# RUN: llvm-readobj -file-headers -r %t2 | FileCheck %s --check-prefix=NOPIE
# NOPIE-NOT: Type: SharedObject

.globl _start
_start:
