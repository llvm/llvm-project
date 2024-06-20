# REQUIRES: loongarch

# RUN: llvm-mc -filetype=obj -triple=loongarch32 %s -o %t.o
# RUN: ld.lld %t.o -o %t
# RUN: llvm-readobj --file-headers %t | FileCheck --check-prefix=LA32 %s
# RUN: ld.lld -m elf32loongarch %t.o -o %t
# RUN: llvm-readobj --file-headers %t | FileCheck --check-prefix=LA32 %s
# RUN: echo 'OUTPUT_FORMAT(elf32-loongarch)' > %t.script
# RUN: ld.lld %t.script %t.o -o %t
# RUN: llvm-readobj --file-headers %t | FileCheck --check-prefix=LA32 %s

# LA32:      ElfHeader {
# LA32-NEXT:   Ident {
# LA32-NEXT:     Magic: (7F 45 4C 46)
# LA32-NEXT:     Class: 32-bit (0x1)
# LA32-NEXT:     DataEncoding: LittleEndian (0x1)
# LA32-NEXT:     FileVersion: 1
# LA32-NEXT:     OS/ABI: SystemV (0x0)
# LA32-NEXT:     ABIVersion: 0
# LA32-NEXT:     Unused: (00 00 00 00 00 00 00)
# LA32-NEXT:   }
# LA32-NEXT:   Type: Executable (0x2)
# LA32-NEXT:   Machine: EM_LOONGARCH (0x102)
# LA32-NEXT:   Version: 1
# LA32-NEXT:   Entry:
# LA32-NEXT:   ProgramHeaderOffset: 0x34
# LA32-NEXT:   SectionHeaderOffset:
# LA32-NEXT:   Flags [ (0x41)
# LA32-NEXT:     EF_LOONGARCH_ABI_SOFT_FLOAT (0x1)
# LA32-NEXT:     EF_LOONGARCH_OBJABI_V1 (0x40)
# LA32-NEXT:   ]
# LA32-NEXT:   HeaderSize: 52
# LA32-NEXT:   ProgramHeaderEntrySize: 32
# LA32-NEXT:   ProgramHeaderCount:
# LA32-NEXT:   SectionHeaderEntrySize: 40
# LA32-NEXT:   SectionHeaderCount:
# LA32-NEXT:   StringTableSectionIndex:
# LA32-NEXT: }

# RUN: llvm-mc -filetype=obj -triple=loongarch64 %s -o %t.o
# RUN: ld.lld %t.o -o %t
# RUN: llvm-readobj --file-headers %t | FileCheck --check-prefix=LA64 %s
# RUN: ld.lld -m elf64loongarch %t.o -o %t
# RUN: llvm-readobj --file-headers %t | FileCheck --check-prefix=LA64 %s
# RUN: echo 'OUTPUT_FORMAT(elf64-loongarch)' > %t.script
# RUN: ld.lld %t.script %t.o -o %t
# RUN: llvm-readobj --file-headers %t | FileCheck --check-prefix=LA64 %s

# LA64:      ElfHeader {
# LA64-NEXT:   Ident {
# LA64-NEXT:     Magic: (7F 45 4C 46)
# LA64-NEXT:     Class: 64-bit (0x2)
# LA64-NEXT:     DataEncoding: LittleEndian (0x1)
# LA64-NEXT:     FileVersion: 1
# LA64-NEXT:     OS/ABI: SystemV (0x0)
# LA64-NEXT:     ABIVersion: 0
# LA64-NEXT:     Unused: (00 00 00 00 00 00 00)
# LA64-NEXT:   }
# LA64-NEXT:   Type: Executable (0x2)
# LA64-NEXT:   Machine: EM_LOONGARCH (0x102)
# LA64-NEXT:   Version: 1
# LA64-NEXT:   Entry:
# LA64-NEXT:   ProgramHeaderOffset: 0x40
# LA64-NEXT:   SectionHeaderOffset:
# LA64-NEXT:   Flags [ (0x43)
# LA64-NEXT:     EF_LOONGARCH_ABI_DOUBLE_FLOAT (0x3)
# LA64-NEXT:     EF_LOONGARCH_OBJABI_V1 (0x40)
# LA64-NEXT:   ]
# LA64-NEXT:   HeaderSize: 64
# LA64-NEXT:   ProgramHeaderEntrySize: 56
# LA64-NEXT:   ProgramHeaderCount:
# LA64-NEXT:   SectionHeaderEntrySize: 64
# LA64-NEXT:   SectionHeaderCount:
# LA64-NEXT:   StringTableSectionIndex:
# LA64-NEXT: }

.globl _start
_start:
