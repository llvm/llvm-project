## Verify that lld defines __eh_frame_start/end and __eh_frame_hdr_start/end symbols

# REQUIRES: x86

# RUN: llvm-mc -filetype=obj -triple=x86_64-unknown-linux %s -o %t.o
# RUN: ld.lld %t.o -eh-frame-hdr -o %t
# RUN: llvm-readelf -s %t | FileCheck %s --check-prefix=SYM

# SYM: __eh_frame_start
# SYM: __eh_frame_end
# SYM: __eh_frame_hdr_start
# SYM: __eh_frame_hdr_end

.text
.globl _start
.type _start, @function
_start:
  nop

# Emit references so the link will fail if these are not defined
check_symbol:
  .quad __eh_frame_start
  .quad __eh_frame_end
  .quad __eh_frame_hdr_start
  .quad __eh_frame_hdr_end
