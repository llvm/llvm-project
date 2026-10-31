# REQUIRES: x86
## CFI jump table relaxation moves f.cfi. .eh_frame_hdr uses the new address.

# RUN: llvm-mc -filetype=obj -triple=x86_64 %s -o %t.o
# RUN: ld.lld --branch-to-branch --eh-frame-hdr %t.o -o %t
# RUN: llvm-readelf -s --unwind %t | FileCheck %s

# CHECK:      entry 0 {
# CHECK-NEXT:   initial_location: 0x[[#%x,F:]]
# CHECK:      [[#%.16x,F]] {{.*}} f.cfi{{$}}

.section .text.jt,"ax",@llvm_cfi_jump_table,8
.globl _start
_start:
  jmp f.cfi
  .balign 8, 0xcc

.section .text.f,"ax",@progbits
f.cfi:
  .cfi_startproc
  ret
  .cfi_endproc
