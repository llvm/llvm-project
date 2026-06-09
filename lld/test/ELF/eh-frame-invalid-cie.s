# REQUIRES: x86
## Test CIE structure errors in .eh_frame.

# RUN: rm -rf %t && split-file %s %t && cd %t
# RUN: llvm-mc -filetype=obj -triple=x86_64 too-small.s -o too-small.o
# RUN: llvm-mc -filetype=obj -triple=x86_64 unexpected-end.s -o unexpected-end.o
# RUN: llvm-mc -filetype=obj -triple=x86_64 failed-string.s -o failed-string.o
# RUN: llvm-mc -filetype=obj -triple=x86_64 failed-leb128.s -o failed-leb128.o

# RUN: not ld.lld --eh-frame-hdr too-small.o 2>&1 | FileCheck %s --check-prefix=TOO-SMALL --implicit-check-not=error:
# RUN: not ld.lld --eh-frame-hdr unexpected-end.o 2>&1 | FileCheck %s --check-prefix=UNEXPECTED-END
# RUN: not ld.lld --eh-frame-hdr failed-string.o 2>&1 | FileCheck %s --check-prefix=FAILED-STRING --implicit-check-not=error:
# RUN: not ld.lld --eh-frame-hdr failed-leb128.o 2>&1 | FileCheck %s --check-prefix=FAILED-LEB128

# TOO-SMALL:      error: corrupted .eh_frame: CIE is too small
# TOO-SMALL-NEXT: >>> defined in too-small.o:(.eh_frame+0x0)

# UNEXPECTED-END:      error: corrupted .eh_frame: unexpected end of CIE
# UNEXPECTED-END-NEXT: >>> defined in unexpected-end.o:(.eh_frame+0x8)

# FAILED-STRING:      error: corrupted .eh_frame: corrupted CIE (failed to read string)
# FAILED-STRING-NEXT: >>> defined in failed-string.o:(.eh_frame+0x9)

# FAILED-LEB128:      error: corrupted .eh_frame: corrupted CIE (failed to read LEB128)
# FAILED-LEB128-NEXT: >>> defined in failed-leb128.o:(.eh_frame+0xc)

#--- too-small.s
.section .eh_frame,"a",@unwind
  .byte 0x03
  .byte 0x00
  .byte 0x00
  .byte 0x00
  .byte 0x00
  .byte 0x00
  .byte 0x00

#--- unexpected-end.s
.section .eh_frame,"a",@unwind
  .byte 0x04
  .byte 0x00
  .byte 0x00
  .byte 0x00
  .byte 0x00
  .byte 0x00
  .byte 0x00
  .byte 0x00

#--- failed-string.s
.section .eh_frame,"a",@unwind
.align 1
  .byte 0x08
  .byte 0x00
  .byte 0x00
  .byte 0x00
  .byte 0x00
  .byte 0x00
  .byte 0x00
  .byte 0x00
  .byte 0x01
  .byte 0x01
  .byte 0x01
  .byte 0x01

#--- failed-leb128.s
.section .eh_frame,"a",@unwind
  .byte 0x08
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
