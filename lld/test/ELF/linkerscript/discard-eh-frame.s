# REQUIRES: x86
## Test discarding .eh_frame and/or .eh_frame_hdr.

# RUN: rm -rf %t && split-file %s %t && cd %t
# RUN: llvm-mc -filetype=obj -triple=x86_64 a.s -o a.o
# RUN: ld.lld --eh-frame-hdr -T 1.lds a.o -o out1
# RUN: llvm-readelf -Sl out1 | FileCheck %s
# RUN: ld.lld --eh-frame-hdr -T 2.lds a.o -o out2
# RUN: llvm-readelf -Sl out2 | FileCheck %s
# RUN: ld.lld --eh-frame-hdr -T 3.lds a.o -o out3
# RUN: llvm-readelf -Sl out3 | FileCheck %s --check-prefix=CHECK3

# CHECK-NOT: .eh_frame
# CHECK:     .text
# CHECK-NOT: .eh_frame

# CHECK:     Program Headers:
# CHECK-NOT: PT_GNU_EH_FRAME

# CHECK3-NOT: .eh_frame_hdr
# CHECK3:     .eh_frame PROGBITS
# CHECK3-NOT: .eh_frame_hdr
# CHECK3:     Program Headers:
# CHECK3-NOT: PT_GNU_EH_FRAME

#--- a.s
.global _start
_start:
 nop

.section .dah,"ax",@progbits
.cfi_startproc
 nop
.cfi_endproc

#--- 1.lds
## Regression test for https://github.com/llvm/llvm-project/pull/179089#issuecomment-3888507749
SECTIONS { /DISCARD/ : { *(.eh_frame) } }

#--- 2.lds
SECTIONS { /DISCARD/ : { *(.eh_frame .eh_frame_hdr) } }

#--- 3.lds
SECTIONS { /DISCARD/ : { *(.eh_frame_hdr) } }
