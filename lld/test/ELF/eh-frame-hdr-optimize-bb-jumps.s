# REQUIRES: x86
## Check that .eh_frame_hdr search-table keys match the final PCs of the
## FDEs they reference after --optimize-bb-jumps.

# RUN: llvm-mc -filetype=obj -triple=x86_64 %s -o %t.o
# RUN: ld.lld --eh-frame-hdr --optimize-bb-jumps %t.o -o %t
# RUN: llvm-readelf --unwind %t | FileCheck %s

# CHECK:      fde_count: 2
# CHECK:      entry 0 {
# CHECK-NEXT:   initial_location: [[PC0:0x[0-9a-f]+]]
# CHECK-NEXT:   address: [[FDE0:0x[0-9a-f]+]]
# CHECK:      entry 1 {
# CHECK-NEXT:   initial_location: [[PC1:0x[0-9a-f]+]]
# CHECK-NEXT:   address: [[FDE1:0x[0-9a-f]+]]
# CHECK:      [{{ *}}[[FDE0]]{{ *}}] FDE
# CHECK-NEXT:   initial_location: [[PC0]]
# CHECK:      [{{ *}}[[FDE1]]{{ *}}] FDE
# CHECK-NEXT:   initial_location: [[PC1]]

.section .text,"ax",@progbits,unique,1
.type foo,@function
foo:
  nop
  jmp a.BB.foo

.section .text,"ax",@progbits,unique,2
.type a.BB.foo,@function
a.BB.foo:
  .cfi_startproc
  nop
  .cfi_endproc

.section .text,"ax",@progbits,unique,3
.type r.BB.foo,@function
r.BB.foo:
  .cfi_startproc
  nop
  .cfi_endproc
