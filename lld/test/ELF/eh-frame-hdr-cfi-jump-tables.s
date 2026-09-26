# REQUIRES: x86
## Check that .eh_frame_hdr search-table keys match the final PCs of the
## FDEs they reference after CFI jump-table relaxation (issue #226166).

# RUN: llvm-mc -filetype=obj -triple=x86_64 %s -o %t.o
# RUN: ld.lld --eh-frame-hdr -O2 %t.o -shared -o %t
# RUN: llvm-readelf --unwind %t | FileCheck %s
# RUN: ld.lld --eh-frame-hdr -O1 --branch-to-branch %t.o -shared -o %t2
# RUN: llvm-readelf --unwind %t2 | FileCheck %s

# CHECK:      fde_count: 1
# CHECK:      entry 0 {
# CHECK-NEXT:   initial_location: [[PC:0x[0-9a-f]+]]
# CHECK-NEXT:   address: [[FDE:0x[0-9a-f]+]]
# CHECK:      [{{ *}}[[FDE]]{{ *}}] FDE
# CHECK-NEXT:   initial_location: [[PC]]

.section .text.jt,"ax",@llvm_cfi_jump_table,8
.type f1,@function
f1:
  jmp f1.cfi
  .balign 8, 0xcc
.type f2,@function
f2:
  jmp f2.cfi
  .balign 8, 0xcc
.type f3,@function
f3:
  jmp f3.cfi
  .balign 8, 0xcc
.type f4,@function
f4:
  jmp f4.cfi
  .balign 8, 0xcc

.section .text.f1,"ax",@progbits
f1.cfi:
  ret
  nop
  nop
  nop
  nop
  nop
  nop
  nop
  nop
  nop

.section .text.f2,"ax",@progbits
f2.cfi:
  ret
  nop
  nop
  nop
  nop
  nop
  nop
  nop
  nop
  nop

.section .text.f3,"ax",@progbits
f3.cfi:
  ret
  nop
  nop
  nop
  nop
  nop
  nop
  nop
  nop
  nop

.section .text.f4,"ax",@progbits
.type f4.cfi,@function
f4.cfi:
  .cfi_startproc
  ret
  .cfi_endproc
.size f4.cfi, .-f4.cfi
