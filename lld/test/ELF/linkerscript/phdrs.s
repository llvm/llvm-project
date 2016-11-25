# REQUIRES: x86
# RUN: llvm-mc -filetype=obj -triple=x86_64-unknown-linux %s -o %t
# RUN: echo "PHDRS {all PT_LOAD FILEHDR PHDRS ;} \
# RUN:       SECTIONS { \
# RUN:           . = 0x10000200; \
# RUN:           .text : {*(.text*)} :all \
# RUN:           .foo : {*(.foo.*)} :all \
# RUN:           .data : {*(.data.*)} :all}" > %t.script
# RUN: ld.lld -o %t1 --script %t.script %t
# RUN: llvm-readobj -program-headers %t1 | FileCheck %s

## Check that program headers are not written, unless we explicitly tell
## lld to do this.
# RUN: echo "PHDRS {all PT_LOAD;} \
# RUN:       SECTIONS { \
# RUN:           . = 0x10000200; \
# RUN:           /DISCARD/ : {*(.text*)}  \
# RUN:           .foo : {*(.foo.*)} :all \
# RUN:       }" > %t.script
# RUN: ld.lld -o %t1 --script %t.script %t
# RUN: llvm-readobj -program-headers %t1 | FileCheck --check-prefix=NOPHDR %s

## Check the AT(expr)
# RUN: echo "PHDRS {all PT_LOAD FILEHDR PHDRS AT(0x500 + 0x500) ;} \
# RUN:       SECTIONS { \
# RUN:           . = 0x10000200; \
# RUN:           .text : {*(.text*)} :all \
# RUN:           .foo : {*(.foo.*)} :all \
# RUN:           .data : {*(.data.*)} :all}" > %t.script
# RUN: ld.lld -o %t1 --script %t.script %t
# RUN: llvm-readobj -program-headers %t1 | FileCheck --check-prefix=AT %s

# RUN: echo "PHDRS {all PT_LOAD FILEHDR PHDRS ;} \
# RUN:       SECTIONS { \
# RUN:           . = 0x10000200; \
# RUN:           .text : {*(.text*)} :all \
# RUN:           .foo : {*(.foo.*)}  \
# RUN:           .data : {*(.data.*)} }" > %t.script
# RUN: ld.lld -o %t1 --script %t.script %t
# RUN: llvm-readobj -program-headers %t1 | FileCheck --check-prefix=DEFHDR %s

# CHECK:     ProgramHeaders [
# CHECK-NEXT:  ProgramHeader {
# CHECK-NEXT:    Type: PT_LOAD (0x1)
# CHECK-NEXT:    Offset: 0x0
# CHECK-NEXT:    VirtualAddress: 0x10000000
# CHECK-NEXT:    PhysicalAddress: 0x10000000
# CHECK-NEXT:    FileSize: 521
# CHECK-NEXT:    MemSize: 521
# CHECK-NEXT:    Flags [ (0x7)
# CHECK-NEXT:      PF_R (0x4)
# CHECK-NEXT:      PF_W (0x2)
# CHECK-NEXT:      PF_X (0x1)
# CHECK-NEXT:    ]

# NOPHDR:     ProgramHeaders [
# NOPHDR-NEXT:  ProgramHeader {
# NOPHDR-NEXT:    Type: PT_LOAD (0x1)
# NOPHDR-NEXT:    Offset: 0x200
# NOPHDR-NEXT:    VirtualAddress: 0x10000200
# NOPHDR-NEXT:    PhysicalAddress: 0x10000200
# NOPHDR-NEXT:    FileSize: 8
# NOPHDR-NEXT:    MemSize: 8
# NOPHDR-NEXT:    Flags [ (0x6)
# NOPHDR-NEXT:      PF_R (0x4)
# NOPHDR-NEXT:      PF_W (0x2)
# NOPHDR-NEXT:    ]
# NOPHDR-NEXT:    Alignment: 4096
# NOPHDR-NEXT:  }
# NOPHDR-NEXT: ]

# AT:       ProgramHeaders [
# AT-NEXT:    ProgramHeader {
# AT-NEXT:      Type: PT_LOAD (0x1)
# AT-NEXT:      Offset: 0x0
# AT-NEXT:      VirtualAddress: 0x10000000
# AT-NEXT:      PhysicalAddress: 0xA00
# AT-NEXT:      FileSize: 521
# AT-NEXT:      MemSize: 521
# AT-NEXT:      Flags [ (0x7)
# AT-NEXT:        PF_R (0x4)
# AT-NEXT:        PF_W (0x2)
# AT-NEXT:        PF_X (0x1)
# AT-NEXT:      ]

## Check the numetic values for PHDRS.
# RUN: echo "PHDRS {text PT_LOAD FILEHDR PHDRS; foo 0x11223344; } \
# RUN:       SECTIONS { . = SIZEOF_HEADERS; .foo : { *(.*) } : text : foo}" > %t1.script
# RUN: ld.lld -o %t2 --script %t1.script %t
# RUN: llvm-readobj -program-headers %t2 | FileCheck --check-prefix=INT-PHDRS %s

# INT-PHDRS:      ProgramHeaders [
# INT-PHDRS:        ProgramHeader {
# INT-PHDRS:           Type:  (0x11223344)
# INT-PHDRS-NEXT:      Offset: 0xB0
# INT-PHDRS-NEXT:      VirtualAddress: 0xB0
# INT-PHDRS-NEXT:      PhysicalAddress: 0xB0
# INT-PHDRS-NEXT:      FileSize: 9
# INT-PHDRS-NEXT:      MemSize: 9
# INT-PHDRS-NEXT:      Flags [
# INT-PHDRS-NEXT:        PF_R
# INT-PHDRS-NEXT:        PF_W
# INT-PHDRS-NEXT:        PF_X
# INT-PHDRS-NEXT:      ]
# INT-PHDRS-NEXT:      Alignment:
# INT-PHDRS-NEXT:    }
# INT-PHDRS-NEXT:  ]

# DEFHDR:     ProgramHeaders [
# DEFHDR-NEXT:  ProgramHeader {
# DEFHDR-NEXT:    Type: PT_LOAD (0x1)
# DEFHDR-NEXT:    Offset: 0x0
# DEFHDR-NEXT:    VirtualAddress: 0x10000000
# DEFHDR-NEXT:    PhysicalAddress: 0x10000000
# DEFHDR-NEXT:    FileSize: 521
# DEFHDR-NEXT:    MemSize: 521
# DEFHDR-NEXT:    Flags [ (0x7)
# DEFHDR-NEXT:      PF_R (0x4)
# DEFHDR-NEXT:      PF_W (0x2)
# DEFHDR-NEXT:      PF_X (0x1)
# DEFHDR-NEXT:    ]

.global _start
_start:
 nop

.section .foo.1,"a"
foo1:
 .long 0

.section .foo.2,"aw"
foo2:
 .long 0
