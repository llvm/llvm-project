// RUN: llvm-mc -filetype=obj -triple=x86_64-unknown-linux %s -o %t.o
// RUN: llvm-mc -filetype=obj -triple=x86_64-unknown-linux %p/Inputs/shared.s -o %t2.o
// RUN: ld.lld -shared %t2.o -o %t2.so
// RUN: ld.lld %t.o %t2.so -z now -z relro -o %t
// RUN: llvm-readobj --program-headers --dynamic-table -t -s -dyn-symbols -section-data %t | FileCheck --check-prefix=FULLRELRO %s
// RUN: ld.lld %t.o %t2.so -z relro -o %t
// RUN: llvm-readobj --program-headers --dynamic-table -t -s -dyn-symbols -section-data %t | FileCheck --check-prefix=PARTRELRO %s
// RUN: ld.lld %t.o %t2.so -z norelro -o %t
// RUN: llvm-readobj --program-headers --dynamic-table -t -s -dyn-symbols -section-data %t | FileCheck --check-prefix=NORELRO %s
// REQUIRES: x86

// FULLRELRO:        Section {
// FULLRELRO:        Index: 9
// FULLRELRO-NEXT:   Name: .got
// FULLRELRO-NEXT:   Type: SHT_PROGBITS
// FULLRELRO-NEXT:   Flags [
// FULLRELRO-NEXT:     SHF_ALLOC
// FULLRELRO-NEXT:     SHF_WRITE
// FULLRELRO-NEXT:   ]
// FULLRELRO-NEXT:   Address: 0x12100
// FULLRELRO-NEXT:   Offset: 0x2100
// FULLRELRO-NEXT:   Size: 8
// FULLRELRO-NEXT:   Link: 0
// FULLRELRO-NEXT:   Info: 0
// FULLRELRO-NEXT:   AddressAlignment: 8
// FULLRELRO-NEXT:   EntrySize: 0
// FULLRELRO-NEXT:   SectionData (
// FULLRELRO-NEXT:     0000: 00000000 00000000
// FULLRELRO-NEXT:   )
// FULLRELRO-NEXT: }
// FULLRELRO-NEXT: Section {
// FULLRELRO-NEXT:   Index: 10
// FULLRELRO-NEXT:   Name: .got.plt
// FULLRELRO-NEXT:   Type: SHT_PROGBITS
// FULLRELRO-NEXT:   Flags [
// FULLRELRO-NEXT:     SHF_ALLOC
// FULLRELRO-NEXT:     SHF_WRITE
// FULLRELRO-NEXT:   ]
// FULLRELRO-NEXT:   Address: 0x12108
// FULLRELRO-NEXT:   Offset: 0x2108
// FULLRELRO-NEXT:   Size: 32
// FULLRELRO-NEXT:   Link: 0
// FULLRELRO-NEXT:   Info: 0
// FULLRELRO-NEXT:   AddressAlignment: 8
// FULLRELRO-NEXT:   EntrySize: 0
// FULLRELRO-NEXT:   SectionData (
// FULLRELRO-NEXT:     0000:
// FULLRELRO-NEXT:     0010:
// FULLRELRO-NEXT:   )
// FULLRELRO-NEXT: }
// FULLRELRO-NEXT: Section {
// FULLRELRO-NEXT:   Index: 11
// FULLRELRO-NEXT:   Name: .data
// FULLRELRO-NEXT:   Type: SHT_PROGBITS
// FULLRELRO-NEXT:   Flags [
// FULLRELRO-NEXT:     SHF_ALLOC
// FULLRELRO-NEXT:     SHF_WRITE
// FULLRELRO-NEXT:   ]
// FULLRELRO-NEXT:   Address: 0x12128
// FULLRELRO-NEXT:   Offset: 0x2128
// FULLRELRO-NEXT:   Size: 12
// FULLRELRO-NEXT:   Link: 0
// FULLRELRO-NEXT:   Info: 0
// FULLRELRO-NEXT:   AddressAlignment:
// FULLRELRO-NEXT:   EntrySize: 0
// FULLRELRO-NEXT:   SectionData (
// FULLRELRO-NEXT:     0000:
// FULLRELRO-NEXT:   )
// FULLRELRO-NEXT: }
// FULLRELRO-NEXT: Section {
// FULLRELRO-NEXT:   Index: 12
// FULLRELRO-NEXT:   Name: .foo
// FULLRELRO-NEXT:   Type: SHT_PROGBITS
// FULLRELRO-NEXT:   Flags [
// FULLRELRO-NEXT:     SHF_ALLOC
// FULLRELRO-NEXT:     SHF_WRITE
// FULLRELRO-NEXT:   ]
// FULLRELRO-NEXT:   Address: 0x12134
// FULLRELRO-NEXT:   Offset: 0x2134
// FULLRELRO-NEXT:   Size: 0
// FULLRELRO-NEXT:   Link: 0
// FULLRELRO-NEXT:   Info: 0
// FULLRELRO-NEXT:   AddressAlignment:
// FULLRELRO-NEXT:   EntrySize: 0
// FULLRELRO-NEXT:   SectionData (
// FULLRELRO-NEXT:   )
// FULLRELRO-NEXT: }
// 308 - sizeof(.data)(12) = 296
// FULLRELRO:       ProgramHeaders [
// FULLRELRO:       Type: PT_LOAD
// FULLRELRO:       Offset: 0x2000
// FULLRELRO-NEXT:  VirtualAddress: [[RWADDR:.*]]
// FULLRELRO-NEXT:  PhysicalAddress:
// FULLRELRO-NEXT:  FileSize: 308
// FULLRELRO-NEXT:  MemSize: 308
// FULLRELRO-NEXT:  Flags [
// FULLRELRO-NEXT:    PF_R
// FULLRELRO-NEXT:    PF_W
// FULLRELRO-NEXT:  ]
// FULLRELRO-NEXT:  Alignment: 4096
// FULLRELRO-NEXT:}
// FULLRELRO:       Type: PT_GNU_RELRO
// FULLRELRO-NEXT:  Offset: 0x
// FULLRELRO-NEXT:  VirtualAddress: [[RWADDR]]
// FULLRELRO-NEXT:  PhysicalAddress:
// FULLRELRO-NEXT:  FileSize: 296
// FULLRELRO-NEXT:  MemSize: 296
// FULLRELRO-NEXT:  Flags [
// FULLRELRO-NEXT:    PF_R
// FULLRELRO-NEXT:  ]
// FULLRELRO-NEXT:  Alignment: 1
// FULLRELRO-NEXT:}

// PARTRELRO:       Section {
// PARTRELRO:       Index: 9
// PARTRELRO-NEXT:  Name: .got
// PARTRELRO-NEXT:  Type: SHT_PROGBITS
// PARTRELRO-NEXT:  Flags [
// PARTRELRO-NEXT:    SHF_ALLOC
// PARTRELRO-NEXT:    SHF_WRITE
// PARTRELRO-NEXT:  ]
// PARTRELRO-NEXT:  Address: 0x120E0
// PARTRELRO-NEXT:  Offset: 0x20E0
// PARTRELRO-NEXT:  Size: 8
// PARTRELRO-NEXT:  Link: 0
// PARTRELRO-NEXT:  Info: 0
// PARTRELRO-NEXT:  AddressAlignment: 8
// PARTRELRO-NEXT:  EntrySize: 0
// PARTRELRO-NEXT:  SectionData (
// PARTRELRO-NEXT:    0000:
// PARTRELRO-NEXT:  )
// PARTRELRO-NEXT:  }
// PARTRELRO-NEXT:  Section {
// PARTRELRO-NEXT:  Index: 10
// PARTRELRO-NEXT:  Name: .data
// PARTRELRO-NEXT:  Type: SHT_PROGBITS
// PARTRELRO-NEXT:  Flags [
// PARTRELRO-NEXT:    SHF_ALLOC
// PARTRELRO-NEXT:    SHF_WRITE
// PARTRELRO-NEXT:  ]
// PARTRELRO-NEXT:  Address: 0x120E8
// PARTRELRO-NEXT:  Offset: 0x20E8
// PARTRELRO-NEXT:  Size: 12
// PARTRELRO-NEXT:  Link: 0
// PARTRELRO-NEXT:  Info: 0
// PARTRELRO-NEXT:  AddressAlignment: 1
// PARTRELRO-NEXT:  EntrySize: 0
// PARTRELRO-NEXT:  SectionData (
// PARTRELRO-NEXT:    0000:
// PARTRELRO-NEXT:  )
// PARTRELRO-NEXT:  }
// PARTRELRO-NEXT:  Section {
// PARTRELRO-NEXT:    Index: 11
// PARTRELRO-NEXT:    Name: .foo
// PARTRELRO-NEXT:    Type: SHT_PROGBITS
// PARTRELRO-NEXT:    Flags [
// PARTRELRO-NEXT:      SHF_ALLOC
// PARTRELRO-NEXT:      SHF_WRITE
// PARTRELRO-NEXT:    ]
// PARTRELRO-NEXT:    Address: 0x120F4
// PARTRELRO-NEXT:    Offset: 0x20F4
// PARTRELRO-NEXT:    Size: 0
// PARTRELRO-NEXT:    Link: 0
// PARTRELRO-NEXT:    Info: 0
// PARTRELRO-NEXT:    AddressAlignment: 1
// PARTRELRO-NEXT:    EntrySize: 0
// PARTRELRO-NEXT:    SectionData (
// PARTRELRO-NEXT:    )
// PARTRELRO-NEXT:  }
// PARTRELRO-NEXT:  Section {
// PARTRELRO-NEXT:    Index: 12
// PARTRELRO-NEXT:    Name: .got.plt
// PARTRELRO-NEXT:    Type: SHT_PROGBITS
// PARTRELRO-NEXT:    Flags [
// PARTRELRO-NEXT:      SHF_ALLOC
// PARTRELRO-NEXT:      SHF_WRITE
// PARTRELRO-NEXT:    ]
// PARTRELRO-NEXT:    Address: 0x120F8
// PARTRELRO-NEXT:    Offset: 0x20F8
// PARTRELRO-NEXT:    Size: 32
// PARTRELRO-NEXT:    Link: 0
// PARTRELRO-NEXT:    Info: 0
// PARTRELRO-NEXT:    AddressAlignment: 8
// PARTRELRO-NEXT:    EntrySize: 0
// PARTRELRO-NEXT:    SectionData (
// PARTRELRO-NEXT:      0000:
// PARTRELRO-NEXT:      0010:
// PARTRELRO-NEXT:    )
// PARTRELRO-NEXT:  }
// PARTRELRO-NEXT:  Section {
// PARTRELRO-NEXT:    Index: 13
// PARTRELRO-NEXT:    Name: .bss
// PARTRELRO-NEXT:    Type: SHT_NOBITS
// PARTRELRO-NEXT:    Flags [
// PARTRELRO-NEXT:      SHF_ALLOC
// PARTRELRO-NEXT:      SHF_WRITE
// PARTRELRO-NEXT:    ]
// PARTRELRO-NEXT:    Address: 0x12118
// PARTRELRO-NEXT:    Offset: 0x2118
// PARTRELRO-NEXT:    Size: 0
// PARTRELRO-NEXT:    Link: 0
// PARTRELRO-NEXT:    Info: 0
// PARTRELRO-NEXT:    AddressAlignment: 1
// PARTRELRO-NEXT:    EntrySize: 0
// PARTRELRO-NEXT:  }
// 232 + sizeof(.data)(12) + align(4) + sizeof(.got.plt)(32) = 280
// PARTRELRO:       ProgramHeader {
// PARTRELRO:       Type: PT_LOAD
// PARTRELRO:       Offset: 0x2000
// PARTRELRO-NEXT:  VirtualAddress: [[RWADDR:.*]]
// PARTRELRO-NEXT:  PhysicalAddress:
// PARTRELRO-NEXT:  FileSize: 280
// PARTRELRO-NEXT:  MemSize: 280
// PARTRELRO-NEXT:  Flags [
// PARTRELRO-NEXT:    PF_R (0x4)
// PARTRELRO-NEXT:    PF_W (0x2)
// PARTRELRO-NEXT:  ]
// PARTRELRO-NEXT:  Alignment: 4096
// PARTRELRO:       Type: PT_GNU_RELRO
// PARTRELRO-NEXT:  Offset: 0x2000
// PARTRELRO-NEXT:  VirtualAddress: [[RWADDR]]
// PARTRELRO-NEXT:  PhysicalAddress:
// PARTRELRO-NEXT:  FileSize: 232
// PARTRELRO-NEXT:  MemSize: 232
// PARTRELRO-NEXT:  Flags [
// PARTRELRO-NEXT:    PF_R
// PARTRELRO-NEXT:  ]
// PARTRELRO-NEXT:  Alignment: 1

// NORELRO:     ProgramHeaders [
// NORELRO-NOT: PT_GNU_RELRO

.global _start
_start:
  .long bar
  jmp *bar@GOTPCREL(%rip)

.section .data,"aw"
.quad 0

.zero 4
.section .foo,"aw"
.section .bss,"",@nobits
