# REQUIRES: x86

# RUN: llvm-mc -filetype=obj -triple=x86_64-pc-linux %s -o %t
# RUN: not ld.lld --eh-frame-hdr %t -o /dev/null 2>&1 | FileCheck %s --check-prefix=ERROR --implicit-check-not=error:
# RUN: ld.lld --eh-frame-hdr %t -o /dev/null --noinhibit-exec 2>&1 | FileCheck %s --check-prefix=WARN --implicit-check-not=error:

# ERROR: error: corrupted .eh_frame: unknown .eh_frame augmentation string: {{.+}}
# WARN: warning: corrupted .eh_frame: unknown .eh_frame augmentation string: {{.+}}

.section .eh_frame,"a",@unwind
  .byte 0x0E
  .byte 0x00
  .byte 0x00
  .byte 0x00
  .byte 0x00
  .byte 0x00
  .byte 0x00
  .byte 0x00
  .byte 0x01
  .byte 0x01
  .byte 0x00
  .byte 0x01
  
  .byte 0x01 # LEB128
  .byte 0x01 # LEB128

  .byte 0x01
  .byte 0x01
  .byte 0x01
  .byte 0x01
