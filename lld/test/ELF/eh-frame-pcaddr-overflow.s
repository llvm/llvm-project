# REQUIRES: x86

# RUN: llvm-mc -filetype=obj -triple=x86_64-pc-linux %s -o %t.o
# RUN: not ld.lld --eh-frame-hdr --section-start .text=0x1000000000000000 \
# RUN:   %t.o -o /dev/null 2>&1 | FileCheck %s
# CHECK: error: {{.*}}.o:(.eh_frame): PC address is too large: 2387527121043355528

.text
.global foo
foo:
 nop

.section .eh_frame, "a"
  .long 12   # Size
  .long 0x00 # ID
  .byte 0x01 # Version.
  
  .byte 0x52 # Augmentation string: 'R','\0'
  .byte 0x00
  
  .byte 0x01
  
  .byte 0x01 # LEB128
  .byte 0x01 # LEB128

  .byte 0x00 # DW_EH_PE_absptr

  .byte 0xFF
 
  .long 12  # Size
  .long 0x14 # ID
  .quad foo + 0x1122334455667788
