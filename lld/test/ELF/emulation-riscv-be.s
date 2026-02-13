# REQUIRES: riscv

# RUN: llvm-mc -filetype=obj -triple=riscv32be %s -o %t.o
# RUN: ld.lld %t.o -o %t
# RUN: llvm-readobj --file-headers %t | FileCheck --check-prefix=RV32BE %s
# RUN: ld.lld -m elf32briscv %t.o -o %t
# RUN: llvm-readobj --file-headers %t | FileCheck --check-prefix=RV32BE %s

# RV32BE:      ElfHeader {
# RV32BE-NEXT:   Ident {
# RV32BE-NEXT:     Magic: (7F 45 4C 46)
# RV32BE-NEXT:     Class: 32-bit (0x1)
# RV32BE-NEXT:     DataEncoding: BigEndian (0x2)
# RV32BE-NEXT:     FileVersion: 1
# RV32BE-NEXT:     OS/ABI: SystemV (0x0)
# RV32BE-NEXT:     ABIVersion: 0
# RV32BE-NEXT:     Unused: (00 00 00 00 00 00 00)
# RV32BE-NEXT:   }
# RV32BE-NEXT:   Type: Executable (0x2)
# RV32BE-NEXT:   Machine: EM_RISCV (0xF3)
# RV32BE-NEXT:   Version: 1
# RV32BE-NEXT:   Entry:
# RV32BE-NEXT:   ProgramHeaderOffset: 0x34
# RV32BE-NEXT:   SectionHeaderOffset:
# RV32BE-NEXT:   Flags [ (0x0)
# RV32BE-NEXT:   ]
# RV32BE-NEXT:   HeaderSize: 52
# RV32BE-NEXT:   ProgramHeaderEntrySize: 32
# RV32BE-NEXT:   ProgramHeaderCount:
# RV32BE-NEXT:   SectionHeaderEntrySize: 40
# RV32BE-NEXT:   SectionHeaderCount:
# RV32BE-NEXT:   StringTableSectionIndex:
# RV32BE-NEXT: }

# RUN: llvm-mc -filetype=obj -triple=riscv64be %s -o %t.o
# RUN: ld.lld %t.o -o %t
# RUN: llvm-readobj --file-headers %t | FileCheck --check-prefix=RV64BE %s
# RUN: ld.lld -m elf64briscv %t.o -o %t
# RUN: llvm-readobj --file-headers %t | FileCheck --check-prefix=RV64BE %s

# RV64BE:      ElfHeader {
# RV64BE-NEXT:   Ident {
# RV64BE-NEXT:     Magic: (7F 45 4C 46)
# RV64BE-NEXT:     Class: 64-bit (0x2)
# RV64BE-NEXT:     DataEncoding: BigEndian (0x2)
# RV64BE-NEXT:     FileVersion: 1
# RV64BE-NEXT:     OS/ABI: SystemV (0x0)
# RV64BE-NEXT:     ABIVersion: 0
# RV64BE-NEXT:     Unused: (00 00 00 00 00 00 00)
# RV64BE-NEXT:   }
# RV64BE-NEXT:   Type: Executable (0x2)
# RV64BE-NEXT:   Machine: EM_RISCV (0xF3)
# RV64BE-NEXT:   Version: 1
# RV64BE-NEXT:   Entry:
# RV64BE-NEXT:   ProgramHeaderOffset: 0x40
# RV64BE-NEXT:   SectionHeaderOffset:
# RV64BE-NEXT:   Flags [ (0x0)
# RV64BE-NEXT:   ]
# RV64BE-NEXT:   HeaderSize: 64
# RV64BE-NEXT:   ProgramHeaderEntrySize: 56
# RV64BE-NEXT:   ProgramHeaderCount:
# RV64BE-NEXT:   SectionHeaderEntrySize: 64
# RV64BE-NEXT:   SectionHeaderCount:
# RV64BE-NEXT:   StringTableSectionIndex:
# RV64BE-NEXT: }

.globl _start
_start:
