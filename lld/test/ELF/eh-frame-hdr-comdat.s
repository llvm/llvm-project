# REQUIRES: x86

# RUN: llvm-mc -filetype=obj -triple=x86_64 %s -o %t
# RUN: ld.lld %t %t -o %t2 --eh-frame-hdr
# RUN: llvm-readelf -u --symbols %t2 | FileCheck %s

# CHECK:      EHFrameHeader {
# CHECK-NEXT:   Address:
# CHECK-NEXT:   Offset:
# CHECK-NEXT:   Size: 0x14
# CHECK-NEXT:   Corresponding Section: .eh_frame_hdr
# CHECK-NEXT:   Header {
# CHECK-NEXT:     version: 1
# CHECK-NEXT:     eh_frame_ptr_enc:
# CHECK-NEXT:     fde_count_enc:
# CHECK-NEXT:     table_enc:
# CHECK-NEXT:     eh_frame_ptr:
# CHECK-NEXT:     fde_count: 1
# CHECK-NEXT:     entry 0 {
# CHECK-NEXT:       initial_location: 0x[[# %x, SYM:]]
# CHECK-NEXT:       address: 0x[[# %x, FDE:]]
# CHECK-NEXT:     }
# CHECK-NEXT:   }
# CHECK-NEXT: }

# CHECK:      [0x[[# FDE]]] FDE

# CHECK:      Symbol table
# CHECK-NEXT:    Num:    Value        {{.*}} Name
# CHECK:              {{0+}}[[# SYM]] {{.*}} bar

.section .text.bar,"axG",@progbits,bar,comdat
.global bar
bar:
.cfi_startproc
    ret
.cfi_endproc
