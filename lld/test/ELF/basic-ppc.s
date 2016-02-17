# RUN: llvm-mc -filetype=obj -triple=powerpc-unknown-freebsd %s -o %t
# RUN: ld.lld -discard-all -shared %t -o %t2
# RUN: llvm-readobj -file-headers -sections -section-data -program-headers %t2 | FileCheck %s
# REQUIRES: ppc

# exits with return code 42 on FreeBSD
.text
 li      0,1
 li      3,1
 sc

// CHECK: Format: ELF32-ppc
// CHECK-NEXT: Arch: powerpc
// CHECK-NEXT: AddressSize: 32bit
// CHECK-NEXT: LoadName:
// CHECK-NEXT: ElfHeader {
// CHECK-NEXT:   Ident {
// CHECK-NEXT:     Magic: (7F 45 4C 46)
// CHECK-NEXT:     Class: 32-bit (0x1)
// CHECK-NEXT:     DataEncoding: BigEndian (0x2)
// CHECK-NEXT:     FileVersion: 1
// CHECK-NEXT:     OS/ABI: FreeBSD (0x9)
// CHECK-NEXT:     ABIVersion: 0
// CHECK-NEXT:     Unused: (00 00 00 00 00 00 00)
// CHECK-NEXT:   }
// CHECK-NEXT:   Type: SharedObject (0x3)
// CHECK-NEXT:   Machine: EM_PPC (0x14)
// CHECK-NEXT:   Version: 1
// CHECK-NEXT:   Entry: 0x0
// CHECK-NEXT:   ProgramHeaderOffset: 0x34
// CHECK-NEXT:   SectionHeaderOffset: 0x2084
// CHECK-NEXT:   Flags [ (0x0)
// CHECK-NEXT:   ]
// CHECK-NEXT:   HeaderSize: 52
// CHECK-NEXT:   ProgramHeaderEntrySize: 32
// CHECK-NEXT:   ProgramHeaderCount: 7
// CHECK-NEXT:   SectionHeaderEntrySize: 40
// CHECK-NEXT:   SectionHeaderCount: 9
// CHECK-NEXT:   StringTableSectionIndex: 7
// CHECK-NEXT: }
// CHECK-NEXT: Sections [
// CHECK-NEXT:   Section {
// CHECK-NEXT:     Index: 0
// CHECK-NEXT:     Name:  (0)
// CHECK-NEXT:     Type: SHT_NULL (0x0)
// CHECK-NEXT:     Flags [ (0x0)
// CHECK-NEXT:     ]
// CHECK-NEXT:     Address: 0x0
// CHECK-NEXT:     Offset: 0x0
// CHECK-NEXT:     Size: 0
// CHECK-NEXT:     Link: 0
// CHECK-NEXT:     Info: 0
// CHECK-NEXT:     AddressAlignment: 0
// CHECK-NEXT:     EntrySize: 0
// CHECK-NEXT:     SectionData (
// CHECK-NEXT:     )
// CHECK-NEXT:   }
// CHECK-NEXT:   Section {
// CHECK-NEXT:     Index: 1
// CHECK-NEXT:     Name: .dynsym (1)
// CHECK-NEXT:     Type: SHT_DYNSYM (0xB)
// CHECK-NEXT:     Flags [ (0x2)
// CHECK-NEXT:       SHF_ALLOC (0x2)
// CHECK-NEXT:     ]
// CHECK-NEXT:     Address: 0x114
// CHECK-NEXT:     Offset: 0x114
// CHECK-NEXT:     Size: 16
// CHECK-NEXT:     Link: 3
// CHECK-NEXT:     Info: 1
// CHECK-NEXT:     AddressAlignment: 4
// CHECK-NEXT:     EntrySize: 16
// CHECK-NEXT:     SectionData (
// CHECK-NEXT:       0000: 00000000 00000000 00000000 00000000  |................|
// CHECK-NEXT:     )
// CHECK-NEXT:   }
// CHECK-NEXT:   Section {
// CHECK-NEXT:     Index: 2
// CHECK-NEXT:     Name: .hash (9)
// CHECK-NEXT:     Type: SHT_HASH (0x5)
// CHECK-NEXT:     Flags [ (0x2)
// CHECK-NEXT:       SHF_ALLOC (0x2)
// CHECK-NEXT:     ]
// CHECK-NEXT:     Address: 0x124
// CHECK-NEXT:     Offset: 0x124
// CHECK-NEXT:     Size: 16
// CHECK-NEXT:     Link: 1
// CHECK-NEXT:     Info: 0
// CHECK-NEXT:     AddressAlignment: 4
// CHECK-NEXT:     EntrySize: 4
// CHECK-NEXT:     SectionData (
// CHECK-NEXT:       0000: 00000001 00000001 00000000 00000000  |................|
// CHECK-NEXT:     )
// CHECK-NEXT:   }
// CHECK-NEXT:   Section {
// CHECK-NEXT:     Index: 3
// CHECK-NEXT:     Name: .dynstr (15)
// CHECK-NEXT:     Type: SHT_STRTAB (0x3)
// CHECK-NEXT:     Flags [ (0x2)
// CHECK-NEXT:       SHF_ALLOC (0x2)
// CHECK-NEXT:     ]
// CHECK-NEXT:     Address: 0x134
// CHECK-NEXT:     Offset: 0x134
// CHECK-NEXT:     Size: 1
// CHECK-NEXT:     Link: 0
// CHECK-NEXT:     Info: 0
// CHECK-NEXT:     AddressAlignment: 1
// CHECK-NEXT:     EntrySize: 0
// CHECK-NEXT:     SectionData (
// CHECK-NEXT:       0000: 00                                   |.|
// CHECK-NEXT:     )
// CHECK-NEXT:   }
// CHECK-NEXT:   Section {
// CHECK-NEXT:     Index: 4
// CHECK-NEXT:     Name: .text (23)
// CHECK-NEXT:     Type: SHT_PROGBITS (0x1)
// CHECK-NEXT:     Flags [ (0x6)
// CHECK-NEXT:       SHF_ALLOC (0x2)
// CHECK-NEXT:       SHF_EXECINSTR (0x4)
// CHECK-NEXT:     ]
// CHECK-NEXT:     Address: 0x1000
// CHECK-NEXT:     Offset: 0x1000
// CHECK-NEXT:     Size: 12
// CHECK-NEXT:     Link: 0
// CHECK-NEXT:     Info: 0
// CHECK-NEXT:     AddressAlignment: 4
// CHECK-NEXT:     EntrySize: 0
// CHECK-NEXT:     SectionData (
// CHECK-NEXT:       0000: 38000001 38600001 44000002           |8...8`..D...|
// CHECK-NEXT:     )
// CHECK-NEXT:   }
// CHECK-NEXT:   Section {
// CHECK-NEXT:     Index: 5
// CHECK-NEXT:     Name: .dynamic (29)
// CHECK-NEXT:     Type: SHT_DYNAMIC (0x6)
// CHECK-NEXT:     Flags [ (0x3)
// CHECK-NEXT:       SHF_ALLOC (0x2)
// CHECK-NEXT:       SHF_WRITE (0x1)
// CHECK-NEXT:     ]
// CHECK-NEXT:     Address: 0x2000
// CHECK-NEXT:     Offset: 0x2000
// CHECK-NEXT:     Size: 48
// CHECK-NEXT:     Link: 3
// CHECK-NEXT:     Info: 0
// CHECK-NEXT:     AddressAlignment: 4
// CHECK-NEXT:     EntrySize: 8
// CHECK-NEXT:     SectionData (
// CHECK-NEXT:       0000: 00000006 00000114 0000000B 00000010  |................|
// CHECK-NEXT:       0010: 00000005 00000134 0000000A 00000001  |.......4........|
// CHECK-NEXT:       0020: 00000004 00000124 00000000 00000000  |.......$........|
// CHECK-NEXT:     )
// CHECK-NEXT:   }
// CHECK-NEXT:   Section {
// CHECK-NEXT:     Index: 6
// CHECK-NEXT:     Name: .symtab (38)
// CHECK-NEXT:     Type: SHT_SYMTAB (0x2)
// CHECK-NEXT:     Flags [ (0x0)
// CHECK-NEXT:     ]
// CHECK-NEXT:     Address: 0x0
// CHECK-NEXT:     Offset: 0x2030
// CHECK-NEXT:     Size: 16
// CHECK-NEXT:     Link: 8
// CHECK-NEXT:     Info: 1
// CHECK-NEXT:     AddressAlignment: 4
// CHECK-NEXT:     EntrySize: 16
// CHECK-NEXT:     SectionData (
// CHECK-NEXT:       0000: 00000000 00000000 00000000 00000000  |................|
// CHECK-NEXT:     )
// CHECK-NEXT:   }
// CHECK-NEXT:   Section {
// CHECK-NEXT:     Index: 7
// CHECK-NEXT:     Name: .shstrtab (46)
// CHECK-NEXT:     Type: SHT_STRTAB (0x3)
// CHECK-NEXT:     Flags [ (0x0)
// CHECK-NEXT:     ]
// CHECK-NEXT:     Address: 0x0
// CHECK-NEXT:     Offset: 0x2040
// CHECK-NEXT:     Size: 64
// CHECK-NEXT:     Link: 0
// CHECK-NEXT:     Info: 0
// CHECK-NEXT:     AddressAlignment: 1
// CHECK-NEXT:     EntrySize: 0
// CHECK-NEXT:     SectionData (
// CHECK-NEXT:       0000: 002E6479 6E73796D 002E6861 7368002E  |..dynsym..hash..|
// CHECK-NEXT:       0010: 64796E73 7472002E 74657874 002E6479  |dynstr..text..dy|
// CHECK-NEXT:       0020: 6E616D69 63002E73 796D7461 62002E73  |namic..symtab..s|
// CHECK-NEXT:       0030: 68737472 74616200 2E737472 74616200  |hstrtab..strtab.|
// CHECK-NEXT:     )
// CHECK-NEXT:   }
// CHECK-NEXT:   Section {
// CHECK-NEXT:     Index: 8
// CHECK-NEXT:     Name: .strtab (56)
// CHECK-NEXT:     Type: SHT_STRTAB (0x3)
// CHECK-NEXT:     Flags [ (0x0)
// CHECK-NEXT:     ]
// CHECK-NEXT:     Address: 0x0
// CHECK-NEXT:     Offset: 0x2080
// CHECK-NEXT:     Size: 1
// CHECK-NEXT:     Link: 0
// CHECK-NEXT:     Info: 0
// CHECK-NEXT:     AddressAlignment: 1
// CHECK-NEXT:     EntrySize: 0
// CHECK-NEXT:     SectionData (
// CHECK-NEXT:       0000: 00                                   |.|
// CHECK-NEXT:     )
// CHECK-NEXT:   }
// CHECK-NEXT: ]
// CHECK-NEXT: ProgramHeaders [
// CHECK-NEXT:   ProgramHeader {
// CHECK-NEXT:     Type: PT_PHDR (0x6)
// CHECK-NEXT:     Offset: 0x34
// CHECK-NEXT:     VirtualAddress: 0x34
// CHECK-NEXT:     PhysicalAddress: 0x34
// CHECK-NEXT:     FileSize: 224
// CHECK-NEXT:     MemSize: 224
// CHECK-NEXT:     Flags [ (0x4)
// CHECK-NEXT:       PF_R (0x4)
// CHECK-NEXT:     ]
// CHECK-NEXT:     Alignment: 4
// CHECK-NEXT:   }
// CHECK-NEXT:   ProgramHeader {
// CHECK-NEXT:     Type: PT_LOAD (0x1)
// CHECK-NEXT:     Offset: 0x0
// CHECK-NEXT:     VirtualAddress: 0x0
// CHECK-NEXT:     PhysicalAddress: 0x0
// CHECK-NEXT:     FileSize: 309
// CHECK-NEXT:     MemSize: 309
// CHECK-NEXT:     Flags [ (0x4)
// CHECK-NEXT:       PF_R (0x4)
// CHECK-NEXT:     ]
// CHECK-NEXT:     Alignment: 4096
// CHECK-NEXT:   }
// CHECK-NEXT:   ProgramHeader {
// CHECK-NEXT:     Type: PT_LOAD (0x1)
// CHECK-NEXT:     Offset: 0x1000
// CHECK-NEXT:     VirtualAddress: 0x1000
// CHECK-NEXT:     PhysicalAddress: 0x1000
// CHECK-NEXT:     FileSize: 12
// CHECK-NEXT:     MemSize: 12
// CHECK-NEXT:     Flags [ (0x5)
// CHECK-NEXT:       PF_R (0x4)
// CHECK-NEXT:       PF_X (0x1)
// CHECK-NEXT:     ]
// CHECK-NEXT:     Alignment: 4096
// CHECK-NEXT:   }
// CHECK-NEXT:   ProgramHeader {
// CHECK-NEXT:     Type: PT_LOAD (0x1)
// CHECK-NEXT:     Offset: 0x2000
// CHECK-NEXT:     VirtualAddress: 0x2000
// CHECK-NEXT:     PhysicalAddress: 0x2000
// CHECK-NEXT:     FileSize: 48
// CHECK-NEXT:     MemSize: 48
// CHECK-NEXT:     Flags [ (0x6)
// CHECK-NEXT:       PF_R (0x4)
// CHECK-NEXT:       PF_W (0x2)
// CHECK-NEXT:     ]
// CHECK-NEXT:     Alignment: 4096
// CHECK-NEXT:   }
// CHECK-NEXT:   ProgramHeader {
// CHECK-NEXT:     Type: PT_DYNAMIC (0x2)
// CHECK-NEXT:     Offset: 0x2000
// CHECK-NEXT:     VirtualAddress: 0x2000
// CHECK-NEXT:     PhysicalAddress: 0x2000
// CHECK-NEXT:     FileSize: 48
// CHECK-NEXT:     MemSize: 48
// CHECK-NEXT:     Flags [ (0x6)
// CHECK-NEXT:       PF_R (0x4)
// CHECK-NEXT:       PF_W (0x2)
// CHECK-NEXT:     ]
// CHECK-NEXT:     Alignment: 4
// CHECK-NEXT:   }
// CHECK-NEXT:   ProgramHeader {
// CHECK-NEXT:     Type: PT_GNU_RELRO (0x6474E552)
// CHECK-NEXT:     Offset: 0x2000
// CHECK-NEXT:     VirtualAddress: 0x2000
// CHECK-NEXT:     PhysicalAddress: 0x2000
// CHECK-NEXT:     FileSize: 48
// CHECK-NEXT:     MemSize: 48
// CHECK-NEXT:     Flags [ (0x4)
// CHECK-NEXT:       PF_R (0x4)
// CHECK-NEXT:     ]
// CHECK-NEXT:     Alignment: 1
// CHECK-NEXT:   }
// CHECK-NEXT:   ProgramHeader {
// CHECK-NEXT:     Type: PT_GNU_STACK (0x6474E551)
// CHECK-NEXT:     Offset: 0x0
// CHECK-NEXT:     VirtualAddress: 0x0
// CHECK-NEXT:     PhysicalAddress: 0x0
// CHECK-NEXT:     FileSize: 0
// CHECK-NEXT:     MemSize: 0
// CHECK-NEXT:     Flags [ (0x6)
// CHECK-NEXT:       PF_R (0x4)
// CHECK-NEXT:       PF_W (0x2)
// CHECK-NEXT:     ]
// CHECK-NEXT:     Alignment: 0
// CHECK-NEXT:   }
// CHECK-NEXT: ]
