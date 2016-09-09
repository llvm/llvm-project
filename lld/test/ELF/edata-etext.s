# REQUIRES: x86
# RUN: llvm-mc -filetype=obj -triple=x86_64-pc-linux %s -o %t.o
# RUN: ld.lld %t.o -o %t
# RUN: llvm-readobj -sections -symbols %t | FileCheck %s

## This checks that:
## 1) Address of _etext is the first location after the last read-only loadable segment.
## 2) Address of _edata points to the end of the last non SHT_NOBITS section.
##    That is how gold/bfd do. At the same time specs says: "If the address of _edata is
##    greater than the address of _etext, the address of _end is same as the address
##    of _edata." (https://docs.oracle.com/cd/E53394_01/html/E54766/u-etext-3c.html).
## 3) Address of _end is different from _edata because of 2.
# CHECK:      Section {
# CHECK:         Index: 1
# CHECK:         Name: .text
# CHECK-NEXT:    Type: SHT_PROGBITS
# CHECK-NEXT:    Flags [
# CHECK-NEXT:      SHF_ALLOC
# CHECK-NEXT:      SHF_EXECINSTR
# CHECK-NEXT:    ]
# CHECK-NEXT:    Address: 0x11000
# CHECK-NEXT:    Offset: 0x1000
# CHECK-NEXT:    Size: 1
# CHECK-NEXT:    Link:
# CHECK-NEXT:    Info:
# CHECK-NEXT:    AddressAlignment:
# CHECK-NEXT:    EntrySize: 0
# CHECK-NEXT:  }
# CHECK-NEXT:  Section {
# CHECK-NEXT:    Index: 2
# CHECK-NEXT:    Name: .data
# CHECK-NEXT:    Type: SHT_PROGBITS
# CHECK-NEXT:    Flags [
# CHECK-NEXT:      SHF_ALLOC
# CHECK-NEXT:      SHF_WRITE
# CHECK-NEXT:    ]
# CHECK-NEXT:    Address: 0x12000
# CHECK-NEXT:    Offset: 0x2000
# CHECK-NEXT:    Size: 2
# CHECK-NEXT:    Link:
# CHECK-NEXT:    Info:
# CHECK-NEXT:    AddressAlignment:
# CHECK-NEXT:    EntrySize:
# CHECK-NEXT:  }
# CHECK-NEXT:  Section {
# CHECK-NEXT:    Index: 3
# CHECK-NEXT:    Name: .bss
# CHECK-NEXT:    Type: SHT_NOBITS
# CHECK-NEXT:    Flags [
# CHECK-NEXT:      SHF_ALLOC
# CHECK-NEXT:      SHF_WRITE
# CHECK-NEXT:    ]
# CHECK-NEXT:    Address: 0x12004
# CHECK-NEXT:    Offset: 0x2002
# CHECK-NEXT:    Size: 6
# CHECK-NEXT:    Link:
# CHECK-NEXT:    Info:
# CHECK-NEXT:    AddressAlignment:
# CHECK-NEXT:    EntrySize:
# CHECK-NEXT:  }
# CHECK:      Symbols [
# CHECK-NEXT:  Symbol {
# CHECK-NEXT:    Name:
# CHECK-NEXT:    Value: 0x0
# CHECK-NEXT:    Size: 0
# CHECK-NEXT:    Binding: Local
# CHECK-NEXT:    Type: None
# CHECK-NEXT:    Other: 0
# CHECK-NEXT:    Section: Undefined
# CHECK-NEXT:  }
# CHECK-NEXT:  Symbol {
# CHECK-NEXT:    Name: _edata
# CHECK-NEXT:    Value: 0x12002
# CHECK-NEXT:    Size: 0
# CHECK-NEXT:    Binding: Global
# CHECK-NEXT:    Type: None
# CHECK-NEXT:    Other: 0
# CHECK-NEXT:    Section: Absolute
# CHECK-NEXT:  }
# CHECK-NEXT:  Symbol {
# CHECK-NEXT:    Name: _end
# CHECK-NEXT:    Value: 0x1200A
# CHECK-NEXT:    Size: 0
# CHECK-NEXT:    Binding: Global
# CHECK-NEXT:    Type: None
# CHECK-NEXT:    Other: 0
# CHECK-NEXT:    Section: Absolute
# CHECK-NEXT:  }
# CHECK-NEXT:  Symbol {
# CHECK-NEXT:    Name: _etext
# CHECK-NEXT:    Value: 0x11001
# CHECK-NEXT:    Size: 0
# CHECK-NEXT:    Binding: Global
# CHECK-NEXT:    Type: None
# CHECK-NEXT:    Other: 0
# CHECK-NEXT:    Section: Absolute
# CHECK-NEXT:  }
# CHECK-NEXT:  Symbol {
# CHECK-NEXT:    Name: _start (20)
# CHECK-NEXT:    Value: 0x11000
# CHECK-NEXT:    Size: 0
# CHECK-NEXT:    Binding: Global (0x1)
# CHECK-NEXT:    Type: None (0x0)
# CHECK-NEXT:    Other: 0
# CHECK-NEXT:    Section: .text (0x1)
# CHECK-NEXT:  }
# CHECK-NEXT: ]

.global _start,_end,_etext,_edata
.text
_start:
  nop
.data
  .word 1
.bss
  .align 4
  .space 6
