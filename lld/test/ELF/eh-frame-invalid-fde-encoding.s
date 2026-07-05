# REQUIRES: x86
## Test EhReader::getFdeEncoding errors in .eh_frame.

# RUN: rm -rf %t && split-file %s %t && cd %t
# RUN: llvm-mc -filetype=obj -triple=x86_64 unknown-aug.s -o unknown-aug.o
# RUN: llvm-mc -filetype=obj -triple=x86_64 corrupted.s -o corrupted.o
# RUN: llvm-mc -filetype=obj -triple=x86_64 unknown-fde-encoding.s -o unknown-fde-encoding.o
# RUN: llvm-mc -filetype=obj -triple=x86_64 aligned-encoding.s -o aligned-encoding.o
# RUN: llvm-mc -filetype=obj -triple=x86_64 unknown-size-encoding.s -o unknown-size-encoding.o

# RUN: not ld.lld --eh-frame-hdr unknown-aug.o 2>&1 | FileCheck %s --check-prefix=UNKNOWN-AUG -DPREFIX=error --implicit-check-not=error:
# RUN: ld.lld --eh-frame-hdr unknown-aug.o --noinhibit-exec 2>&1 | FileCheck %s --check-prefix=UNKNOWN-AUG -DPREFIX=warning
# RUN: not ld.lld --eh-frame-hdr corrupted.o 2>&1 | FileCheck %s --check-prefix=CORRUPTED --implicit-check-not=error:
# RUN: not ld.lld --eh-frame-hdr unknown-fde-encoding.o 2>&1 | FileCheck %s --check-prefix=UNKNOWN-FDE --implicit-check-not=error:
# RUN: ld.lld --eh-frame-hdr unknown-fde-encoding.o --noinhibit-exec
# RUN: not ld.lld --eh-frame-hdr aligned-encoding.o 2>&1 | FileCheck %s --check-prefix=ALIGNED --implicit-check-not=error:
# RUN: not ld.lld --eh-frame-hdr unknown-size-encoding.o 2>&1 | FileCheck %s --check-prefix=UNKNOWN-SIZE --implicit-check-not=error:

# UNKNOWN-AUG: [[PREFIX]]: corrupted .eh_frame: unknown .eh_frame augmentation string: {{.+}}

# CORRUPTED: error: corrupted .eh_frame: corrupted CIE

# UNKNOWN-FDE:      error: corrupted .eh_frame: unknown FDE encoding
# UNKNOWN-FDE-NEXT: >>> defined in unknown-fde-encoding.o:(.eh_frame+0xe)

# ALIGNED: error: corrupted .eh_frame: DW_EH_PE_aligned encoding is not supported

# UNKNOWN-SIZE: error: unknown FDE size encoding

#--- unknown-aug.s
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

#--- corrupted.s
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

  .byte 0x50 # Augmentation string: 'P','\0'
  .byte 0x00

  .byte 0x01

  .byte 0x01 # LEB128
  .byte 0x01 # LEB128

  .byte 0x03
  .byte 0x01
  .byte 0x01
  .byte 0x01

#--- unknown-fde-encoding.s
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

  .byte 0x50 # Augmentation string: 'P','\0'
  .byte 0x00

  .byte 0x01

  .byte 0x01 # LEB128
  .byte 0x01 # LEB128

  .byte 0x01
  .byte 0x01
  .byte 0x01
  .byte 0x01

#--- aligned-encoding.s
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

  .byte 0x50 # Augmentation string: 'P','\0'
  .byte 0x00

  .byte 0x01

  .byte 0x01 # LEB128
  .byte 0x01 # LEB128

  .byte 0x51
  .byte 0x01
  .byte 0x01
  .byte 0x01

#--- unknown-size-encoding.s
.section .eh_frame,"a",@unwind
  .long 12   # Size
  .long 0x00 # ID
  .byte 0x01 # Version.

  .byte 0x52 # Augmentation string: 'R','\0'
  .byte 0x00

# Code and data alignment factors.
  .byte 0x01 # LEB128
  .byte 0x01 # LEB128

# Return address register.
  .byte 0x01 # LEB128

  .byte 0xFE # 'R' value: invalid <0xFE>

  .byte 0xFF

  .long 12  # Size
  .long 0x14 # ID
  .quad .eh_frame
