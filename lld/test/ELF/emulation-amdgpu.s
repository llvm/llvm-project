# REQUIRES: amdgpu

# RUN: llvm-mc -filetype=obj -triple=amdgcn-amd-amdhsa %s -o %t.o
# RUN: ld.lld %t.o -o %t
# RUN: llvm-readobj --file-headers %t | FileCheck %s
# RUN: ld.lld -m elf64_amdgpu %t.o -o %t
# RUN: llvm-readobj --file-headers %t | FileCheck %s

# CHECK:      ElfHeader {
# CHECK-NEXT:   Ident {
# CHECK-NEXT:     Magic: (7F 45 4C 46)
# CHECK-NEXT:     Class: 64-bit (0x2)
# CHECK-NEXT:     DataEncoding: LittleEndian (0x1)
# CHECK-NEXT:     FileVersion: 1
# CHECK-NEXT:     OS/ABI: AMDGPU_HSA (0x40)
# CHECK-NEXT:     ABIVersion: 3
# CHECK-NEXT:     Unused: (00 00 00 00 00 00 00)
# CHECK-NEXT:   }
# CHECK-NEXT:   Type: Executable (0x2)
# CHECK-NEXT:   Machine: EM_AMDGPU (0xE0)
# CHECK-NEXT:   Version: 1
# CHECK-NEXT:   Entry:
# CHECK-NEXT:   ProgramHeaderOffset: 0x40
# CHECK-NEXT:   SectionHeaderOffset:
# CHECK-NEXT:   Flags [ (0x0)
# CHECK-NEXT:   ]
# CHECK-NEXT:   HeaderSize: 64
# CHECK-NEXT:   ProgramHeaderEntrySize: 56
# CHECK-NEXT:   ProgramHeaderCount:
# CHECK-NEXT:   SectionHeaderEntrySize: 64
# CHECK-NEXT:   SectionHeaderCount:
# CHECK-NEXT:   StringTableSectionIndex:
# CHECK-NEXT: }

.globl _start
_start:
