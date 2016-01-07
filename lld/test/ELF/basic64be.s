# RUN: llvm-mc -filetype=obj -triple=powerpc64-unknown-linux %s -o %t
# RUN: ld.lld -discard-all %t -o %t2
# RUN: llvm-readobj -file-headers -sections -section-data -program-headers %t2 | FileCheck %s
# REQUIRES: ppc

# exits with return code 42 on linux
.section        ".opd","aw"
.global _start
_start:
.quad   .Lfoo,.TOC.@tocbase,0

# generate .toc and .toc1 sections to make sure that the ordering is as
# intended (.toc before .toc1, and both before .opd).
.section        ".toc1","aw"
.quad          22, 37, 89, 47

.section        ".toc","aw"
.quad          45, 86, 72, 24

.text
.Lfoo:
	li      0,1
	li      3,42
	sc

# CHECK:      ElfHeader {
# CHECK-NEXT:   Ident {
# CHECK-NEXT:     Magic: (7F 45 4C 46)
# CHECK-NEXT:     Class: 64-bit (0x2)
# CHECK-NEXT:     DataEncoding: BigEndian (0x2)
# CHECK-NEXT:     FileVersion: 1
# CHECK-NEXT:     OS/ABI: SystemV (0x0)
# CHECK-NEXT:     ABIVersion: 0
# CHECK-NEXT:     Unused: (00 00 00 00 00 00 00)
# CHECK-NEXT:   }
# CHECK-NEXT:   Type: Executable (0x2)
# CHECK-NEXT:   Machine: EM_PPC64 (0x15)
# CHECK-NEXT:   Version: 1
# CHECK-NEXT:   Entry: 0x10020040
# CHECK-NEXT:   ProgramHeaderOffset: 0x40
# CHECK-NEXT:   SectionHeaderOffset: 0x200C8
# CHECK-NEXT:   Flags [ (0x0)
# CHECK-NEXT:   ]
# CHECK-NEXT:   HeaderSize: 64
# CHECK-NEXT:   ProgramHeaderEntrySize: 56
# CHECK-NEXT:   ProgramHeaderCount: 5
# CHECK-NEXT:   SectionHeaderEntrySize: 64
# CHECK-NEXT:   SectionHeaderCount: 8
# CHECK-NEXT:   StringTableSectionIndex: 6
# CHECK-NEXT: }
# CHECK-NEXT: Sections [
# CHECK-NEXT:   Section {
# CHECK-NEXT:     Index: 0
# CHECK-NEXT:     Name:  (0)
# CHECK-NEXT:     Type: SHT_NULL (0x0)
# CHECK-NEXT:     Flags [ (0x0)
# CHECK-NEXT:     ]
# CHECK-NEXT:     Address: 0x0
# CHECK-NEXT:     Offset: 0x0
# CHECK-NEXT:     Size: 0
# CHECK-NEXT:     Link: 0
# CHECK-NEXT:     Info: 0
# CHECK-NEXT:     AddressAlignment: 0
# CHECK-NEXT:     EntrySize: 0
# CHECK-NEXT:     SectionData (
# CHECK-NEXT:     )
# CHECK-NEXT:   }
# CHECK-NEXT:   Section {
# CHECK-NEXT:     Index: 1
# CHECK-NEXT:     Name: .text (1)
# CHECK-NEXT:     Type: SHT_PROGBITS (0x1)
# CHECK-NEXT:     Flags [ (0x6)
# CHECK-NEXT:       SHF_ALLOC (0x2)
# CHECK-NEXT:       SHF_EXECINSTR (0x4)
# CHECK-NEXT:     ]
# CHECK-NEXT:     Address: 0x10010000
# CHECK-NEXT:     Offset: 0x10000
# CHECK-NEXT:     Size: 12
# CHECK-NEXT:     Link: 0
# CHECK-NEXT:     Info: 0
# CHECK-NEXT:     AddressAlignment: 4
# CHECK-NEXT:     EntrySize: 0
# CHECK-NEXT:     SectionData (
# CHECK:          )
# CHECK-NEXT:   }
# CHECK-NEXT:   Section {
# CHECK-NEXT:     Index: 2
# CHECK-NEXT:     Name: .toc
# CHECK-NEXT:     Type: SHT_PROGBITS (0x1)
# CHECK-NEXT:     Flags [ (0x3)
# CHECK-NEXT:       SHF_ALLOC (0x2)
# CHECK-NEXT:       SHF_WRITE (0x1)
# CHECK-NEXT:     ]
# CHECK-NEXT:     Address: 0x10020000
# CHECK-NEXT:     Offset: 0x20000
# CHECK-NEXT:     Size: 32
# CHECK-NEXT:     Link: 0
# CHECK-NEXT:     Info: 0
# CHECK-NEXT:     AddressAlignment: 1
# CHECK-NEXT:     EntrySize: 0
# CHECK-NEXT:     SectionData (
# CHECK-NEXT:      0000: 00000000 0000002D 00000000 00000056 |.......-.......V|
# CHECK-NEXT:      0010: 00000000 00000048 00000000 00000018 |.......H........|
# CHECK-NEXT:     )
# CHECK-NEXT:   }
# CHECK-NEXT:   Section {
# CHECK-NEXT:     Index: 3
# CHECK-NEXT:     Name: .toc1
# CHECK-NEXT:     Type: SHT_PROGBITS (0x1)
# CHECK-NEXT:     Flags [ (0x3)
# CHECK-NEXT:       SHF_ALLOC (0x2)
# CHECK-NEXT:       SHF_WRITE (0x1)
# CHECK-NEXT:     ]
# CHECK-NEXT:     Address: 0x10020020
# CHECK-NEXT:     Offset: 0x20020
# CHECK-NEXT:     Size: 32
# CHECK-NEXT:     Link: 0
# CHECK-NEXT:     Info: 0
# CHECK-NEXT:     AddressAlignment: 1
# CHECK-NEXT:     EntrySize: 0
# CHECK-NEXT:     SectionData (
# CHECK-NEXT:       0000: 00000000 00000016 00000000 00000025  |...............%|
# CHECK-NEXT:       0010: 00000000 00000059 00000000 0000002F  |.......Y......./|
# CHECK-NEXT:     )
# CHECK-NEXT:   }
# CHECK-NEXT:   Section {
# CHECK-NEXT:     Index: 4
# CHECK-NEXT:     Name: .opd
# CHECK-NEXT:     Type: SHT_PROGBITS (0x1)
# CHECK-NEXT:     Flags [ (0x3)
# CHECK-NEXT:       SHF_ALLOC (0x2)
# CHECK-NEXT:       SHF_WRITE (0x1)
# CHECK-NEXT:     ]
# CHECK-NEXT:     Address: 0x10020040
# CHECK-NEXT:     Offset: 0x20040
# CHECK-NEXT:     Size: 24
# CHECK-NEXT:     Link: 0
# CHECK-NEXT:     Info: 0
# CHECK-NEXT:     AddressAlignment: 1
# CHECK-NEXT:     EntrySize: 0
# CHECK-NEXT:     SectionData (
# CHECK-NEXT:       0000: 00000000 10010000 00000000 00008000 |................|
# CHECK-NEXT:       0010: 00000000 00000000                    |........|
# CHECK-NEXT:     )
# CHECK-NEXT:   }
# CHECK-NEXT:   Section {
# CHECK-NEXT:     Index: 5
# CHECK-NEXT:     Name: .symtab
# CHECK-NEXT:     Type: SHT_SYMTAB (0x2)
# CHECK-NEXT:     Flags [ (0x0)
# CHECK-NEXT:     ]
# CHECK-NEXT:     Address: 0x0
# CHECK-NEXT:     Offset: 0x20058
# CHECK-NEXT:     Size: 48
# CHECK-NEXT:     Link: 7
# CHECK-NEXT:     Info: 1
# CHECK-NEXT:     AddressAlignment: 8
# CHECK-NEXT:     EntrySize: 24
# CHECK-NEXT:     SectionData (
# CHECK:          )
# CHECK-NEXT:   }
# CHECK-NEXT:   Section {
# CHECK-NEXT:     Index: 6
# CHECK-NEXT:     Name: .shstrtab
# CHECK-NEXT:     Type: SHT_STRTAB (0x3)
# CHECK-NEXT:     Flags [ (0x0)
# CHECK-NEXT:     ]
# CHECK-NEXT:     Address: 0x0
# CHECK-NEXT:     Offset: 0x20088
# CHECK-NEXT:     Size: 49
# CHECK-NEXT:     Link: 0
# CHECK-NEXT:     Info: 0
# CHECK-NEXT:     AddressAlignment: 1
# CHECK-NEXT:     EntrySize: 0
# CHECK-NEXT:     SectionData (
# CHECK:          )
# CHECK-NEXT:   }
# CHECK-NEXT:   Section {
# CHECK-NEXT:     Index: 7
# CHECK-NEXT:     Name: .strtab (41)
# CHECK-NEXT:     Type: SHT_STRTAB (0x3)
# CHECK-NEXT:     Flags [ (0x0)
# CHECK-NEXT:     ]
# CHECK-NEXT:     Address: 0x0
# CHECK-NEXT:     Offset: 0x200B9
# CHECK-NEXT:     Size: 8
# CHECK-NEXT:     Link: 0
# CHECK-NEXT:     Info: 0
# CHECK-NEXT:     AddressAlignment: 1
# CHECK-NEXT:     EntrySize: 0
# CHECK-NEXT:     SectionData (
# CHECK-NEXT:       0000: 005F7374 61727400                    |._start.|
# CHECK-NEXT:     )
# CHECK-NEXT:   }
# CHECK-NEXT: ]
# CHECK-NEXT: ProgramHeaders [
# CHECK-NEXT:   ProgramHeader {
# CHECK-NEXT:     Type: PT_PHDR (0x6)
# CHECK-NEXT:     Offset: 0x40
# CHECK-NEXT:     VirtualAddress: 0x10000040
# CHECK-NEXT:     PhysicalAddress: 0x10000040
# CHECK-NEXT:     FileSize: 280
# CHECK-NEXT:     MemSize: 280
# CHECK-NEXT:     Flags [ (0x4)
# CHECK-NEXT:       PF_R (0x4)
# CHECK-NEXT:     ]
# CHECK-NEXT:     Alignment: 8
# CHECK-NEXT:   }
# CHECK-NEXT:  ProgramHeader {
# CHECK-NEXT:    Type: PT_LOAD (0x1)
# CHECK-NEXT:    Offset: 0x0
# CHECK-NEXT:    VirtualAddress: 0x10000000
# CHECK-NEXT:    PhysicalAddress: 0x10000000
# CHECK-NEXT:    FileSize: 344
# CHECK-NEXT:    MemSize: 344
# CHECK-NEXT:    Flags [
# CHECK-NEXT:      PF_R
# CHECK-NEXT:    ]
# CHECK-NEXT:    Alignment: 65536
# CHECK-NEXT:  }
# CHECK-NEXT:  ProgramHeader {
# CHECK-NEXT:    Type: PT_LOAD (0x1)
# CHECK-NEXT:    Offset: 0x10000
# CHECK-NEXT:    VirtualAddress: 0x10010000
# CHECK-NEXT:    PhysicalAddress: 0x10010000
# CHECK-NEXT:    FileSize: 12
# CHECK-NEXT:    MemSize: 12
# CHECK-NEXT:    Flags [ (0x5)
# CHECK-NEXT:      PF_R (0x4)
# CHECK-NEXT:      PF_X (0x1)
# CHECK-NEXT:    ]
# CHECK-NEXT:    Alignment: 65536
# CHECK-NEXT:  }
# CHECK-NEXT:  ProgramHeader {
# CHECK-NEXT:    Type: PT_LOAD (0x1)
# CHECK-NEXT:    Offset: 0x20000
# CHECK-NEXT:    VirtualAddress: 0x10020000
# CHECK-NEXT:    PhysicalAddress: 0x10020000
# CHECK-NEXT:    FileSize: 88
# CHECK-NEXT:    MemSize: 88
# CHECK-NEXT:    Flags [ (0x6)
# CHECK-NEXT:      PF_R (0x4)
# CHECK-NEXT:      PF_W (0x2)
# CHECK-NEXT:    ]
# CHECK-NEXT:    Alignment: 65536
# CHECK-NEXT:  }
# CHECK-NEXT:  ProgramHeader {
# CHECK-NEXT:    Type: PT_GNU_STACK (0x6474E551)
# CHECK-NEXT:    Offset: 0x0
# CHECK-NEXT:    VirtualAddress: 0x0
# CHECK-NEXT:    PhysicalAddress: 0x0
# CHECK-NEXT:    FileSize: 0
# CHECK-NEXT:    MemSize: 0
# CHECK-NEXT:    Flags [ (0x6)
# CHECK-NEXT:      PF_R (0x4)
# CHECK-NEXT:      PF_W (0x2)
# CHECK-NEXT:    ]
# CHECK-NEXT:    Alignment: 0
# CHECK-NEXT:  }
# CHECK-NEXT: ]
