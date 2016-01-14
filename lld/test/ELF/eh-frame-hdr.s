// REQUIRES: x86
// RUN: llvm-mc -filetype=obj -triple=x86_64-pc-linux %s -o %t.o
// RUN: ld.lld %t.o -o %t
// RUN: llvm-readobj -file-headers -s -section-data -program-headers -symbols %t | FileCheck %s --check-prefix=NOHDR
// RUN: ld.lld --eh-frame-hdr %t.o -o %t
// RUN: llvm-readobj -file-headers -s -section-data -program-headers -symbols %t | FileCheck %s --check-prefix=HDR
// RUN: llvm-objdump -d %t | FileCheck %s --check-prefix=HDRDISASM

.section foo,"ax",@progbits
.cfi_startproc
 nop
.cfi_endproc

.section bar,"ax",@progbits
.cfi_startproc
 nop
.cfi_endproc

.section dah,"ax",@progbits
.cfi_startproc
 nop
.cfi_endproc

.text
.globl _start;
_start:

// NOHDR:       Sections [
// NOHDR-NOT:    Name: .eh_frame_hdr
// NOHDR:      ProgramHeaders [
// NOHDR-NOT:   PT_GNU_EH_FRAME

//HDRDISASM:      Disassembly of section foo:
//HDRDISASM-NEXT: foo:
//HDRDISASM-NEXT:    11000: 90 nop
//HDRDISASM-NEXT: Disassembly of section bar:
//HDRDISASM-NEXT: bar:
//HDRDISASM-NEXT:    11001: 90 nop
//HDRDISASM-NEXT: Disassembly of section dah:
//HDRDISASM-NEXT: dah:
//HDRDISASM-NEXT:    11002: 90 nop

// HDR:       Sections [
// HDR:        Section {
// HDR:        Index: 1
// HDR-NEXT:    Name: .eh_frame
// HDR-NEXT:    Type: SHT_X86_64_UNWIND
// HDR-NEXT:    Flags [
// HDR-NEXT:      SHF_ALLOC
// HDR-NEXT:    ]
// HDR-NEXT:    Address: 0x10158
// HDR-NEXT:    Offset: 0x158
// HDR-NEXT:    Size: 96
// HDR-NEXT:    Link: 0
// HDR-NEXT:    Info: 0
// HDR-NEXT:    AddressAlignment: 8
// HDR-NEXT:    EntrySize: 0
// HDR-NEXT:    SectionData (
// HDR-NEXT:      0000: 14000000 00000000 017A5200 01781001  |
// HDR-NEXT:      0010: 1B0C0708 90010000 14000000 1C000000  |
// HDR-NEXT:      0020: 880E0000 01000000 00000000 00000000  |
// HDR-NEXT:      0030: 14000000 34000000 710E0000 01000000  |
// HDR-NEXT:      0040: 00000000 00000000 14000000 4C000000  |
// HDR-NEXT:      0050: 5A0E0000 01000000 00000000 00000000  |
//            CIE: 14000000 00000000 017A5200 01781001 1B0C0708 90010000
//            FDE(1): 14000000 1C000000 880E0000 01000000 00000000 00000000
//                    address of data (starts with 0x880E0000) = 0x10158 + 0x0020 = 0x10178
//                    The starting address to which this FDE applies = 0xE88 + 0x10178 = 0x11000
//                    The number of bytes after the start address to which this FDE applies = 0x01000000 = 1
//            FDE(2): 14000000 34000000 710E0000 01000000 00000000 00000000
//                    address of data (starts with 0x710E0000) = 0x10158 + 0x0038 = 0x10190
//                    The starting address to which this FDE applies = 0xE71 + 0x10190 = 0x11001
//                    The number of bytes after the start address to which this FDE applies = 0x01000000 = 1
//            FDE(3): 14000000 4C000000 5A0E0000 01000000 00000000 00000000
//                    address of data (starts with 0x5A0E0000) = 0x10158 + 0x0050 = 0x101A8
//                    The starting address to which this FDE applies = 0xE5A + 0x101A8 = 0x11002
//                    The number of bytes after the start address to which this FDE applies = 0x01000000 = 1
// HDR-NEXT:    )
// HDR-NEXT:  }
// HDR-NEXT:  Section {
// HDR-NEXT:    Index: 2
// HDR-NEXT:    Name: .eh_frame_hdr
// HDR-NEXT:    Type: SHT_PROGBITS
// HDR-NEXT:    Flags [
// HDR-NEXT:      SHF_ALLOC
// HDR-NEXT:    ]
// HDR-NEXT:    Address: 0x101B8
// HDR-NEXT:    Offset: 0x1B8
// HDR-NEXT:    Size: 36
// HDR-NEXT:    Link: 0
// HDR-NEXT:    Info: 0
// HDR-NEXT:    AddressAlignment: 0
// HDR-NEXT:    EntrySize: 0
// HDR-NEXT:    SectionData (
// HDR-NEXT:      0000: 011B033B 9CFFFFFF 03000000 480E0000  |
// HDR-NEXT:      0010: B8FFFFFF 490E0000 D0FFFFFF 4A0E0000  |
// HDR-NEXT:      0020: E8FFFFFF                             |
//                Header (always 4 bytes): 0x011B033B
//                   9CFFFFFF = .eh_frame(0x10158) - .eh_frame_hdr(0x101B8) - 4
//                   03000000 = 3 = the number of FDE pointers in the table.
//                Entry(1): 480E0000 B8FFFFFF
//                   480E0000 = 0x11000 - .eh_frame_hdr(0x101B8) = 0xE48
//                   B8FFFFFF = address of FDE(1) - .eh_frame_hdr(0x101B8) =
//                      = .eh_frame(0x10158) + 24 - 0x101B8 = 0xFFFFFFB8
//                Entry(2): 490E0000 D0FFFFFF
//                   490E0000 = 0x11001 - .eh_frame_hdr(0x101B8) = 0xE49
//                   D0FFFFFF = address of FDE(2) - .eh_frame_hdr(0x101B8) =
//                      = .eh_frame(0x10158) + 24 + 24 - 0x101B8 = 0xFFFFFFD0
//                Entry(3): 4A0E0000 E8FFFFFF
//                   4A0E0000 = 0x11002 - .eh_frame_hdr(0x101B8) = 0xE4A
//                   E8FFFFFF = address of FDE(2) - .eh_frame_hdr(0x101B8) =
//                      = .eh_frame(0x10158) + 24 + 24 - 0x101B8 = 0xFFFFFFE8
// HDR-NEXT:    )
// HDR-NEXT:  }
// HDR:     ProgramHeaders [
// HDR:      ProgramHeader {
// HDR:       Type: PT_GNU_EH_FRAME
// HDR-NEXT:   Offset: 0x1B8
// HDR-NEXT:   VirtualAddress: 0x101B8
// HDR-NEXT:   PhysicalAddress: 0x101B8
// HDR-NEXT:   FileSize: 36
// HDR-NEXT:   MemSize: 36
// HDR-NEXT:   Flags [
// HDR-NEXT:     PF_R
// HDR-NEXT:   ]
// HDR-NEXT:   Alignment: 1
// HDR-NEXT: }
