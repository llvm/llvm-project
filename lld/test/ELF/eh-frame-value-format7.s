# REQUIRES: x86

# RUN: llvm-mc -filetype=obj -triple=x86_64-pc-linux %s -o %t.o
# RUN: ld.lld --eh-frame-hdr --section-start .text=0x1000 %t.o -o %t
# RUN: llvm-readobj -s -section-data %t | FileCheck %s

## Check we are able to handle DW_EH_PE_udata2 encoding.

# CHECK:        Name: .eh_frame_hdr
# CHECK:      Section {
# CHECK:        Index:
# CHECK:        Name: .eh_frame
# CHECK-NEXT:   Type: SHT_PROGBITS
# CHECK-NEXT:   Flags [
# CHECK-NEXT:     SHF_ALLOC
# CHECK-NEXT:   ]
# CHECK-NEXT:   Address:
# CHECK-NEXT:   Offset:
# CHECK-NEXT:   Size:
# CHECK-NEXT:   Link:
# CHECK-NEXT:   Info:
# CHECK-NEXT:   AddressAlignment:
# CHECK-NEXT:   EntrySize:
# CHECK-NEXT:   SectionData (
# CHECK-NEXT:     0000: 0C000000 00000000 01520001 010102FF
# CHECK-NEXT:     0010: 0C000000 14000000 34120000 00000000
#                                           ^
#                                           ---> ADDR(foo) + 0x234 = 0x1234

.text
.global foo
foo:
 nop

.section .eh_frame
  .long 12   # Size
  .long 0x00 # ID
  .byte 0x01 # Version.
  
  .byte 0x52 # Augmentation string: 'R','\0'
  .byte 0x00
  
  .byte 0x01
  
  .byte 0x01 # LEB128
  .byte 0x01 # LEB128

  .byte 0x02 # DW_EH_PE_udata2

  .byte 0xFF
 
  .long 0x6  # Size
  .long 0x14 # ID
  .short foo + 0x234
