## A fills cluster 0 exactly. The call in B is beyond the modeled 64-byte
## cluster range, but a thunk at the start of cluster 1 is exactly 64 bytes
## from A. The equality case must allow that final backward hop.

# REQUIRES: system-linux

# RUN: %clang %cflags -Wl,-q -Wl,-e,A %s -o %t -nostdlib
# RUN: link_fdata --no-lbr %s %t %t.fdata
# RUN: llvm-strip --strip-unneeded %t
# RUN: llvm-bolt %t -o %t.bolt --data %t.fdata --relax-exp \
# RUN:   --max-cluster-size=64 | FileCheck %s --check-prefix=CHECK-BOLT
# RUN: llvm-objdump -d --disassemble-symbols=A,B,__AArch64_backward_Thunk_A_0 \
# RUN:   %t.bolt | FileCheck %s --check-prefix=CHECK-OUTPUT

# CHECK-BOLT: BOLT-INFO: built 2 function fragment cluster(s)
# CHECK-BOLT: BOLT-INFO: cluster: 0
# CHECK-BOLT-NEXT: BOLT-INFO:   1 fragment(s)
# CHECK-BOLT-NEXT: BOLT-INFO:   64 estimated bytes without thunks
# CHECK-BOLT-NEXT: BOLT-INFO:   0 thunk bytes
# CHECK-BOLT-NEXT: BOLT-INFO: cluster: 1
# CHECK-BOLT-NEXT: BOLT-INFO:   1 fragment(s)
# CHECK-BOLT-NEXT: BOLT-INFO:   24 estimated bytes without thunks
# CHECK-BOLT-NEXT: BOLT-INFO:   4 thunk bytes
# CHECK-BOLT: BOLT-INFO: relaxed 1 calls with short thunks
# CHECK-BOLT: BOLT-INFO: 1 short thunks created

  .text
  .globl A
  .type A, %function
A:
.A_entry:
# FDATA: 1 A #.A_entry# 100
  ret
  .space 0x34
  .size A, .-A

  .globl B
  .type B, %function
B:
.B_entry:
# FDATA: 1 B #.B_entry# 100
  mov x0, #1
  bl A
  ret
  .size B, .-B

## Force relocation mode.
  .reloc 0, R_AARCH64_NONE

# CHECK-OUTPUT:      <__AArch64_backward_Thunk_A_0>:
# CHECK-OUTPUT-NEXT:   b {{.*}} <A>
# CHECK-OUTPUT:      <B>:
# CHECK-OUTPUT:        bl {{.*}} <__AArch64_backward_Thunk_A_0>
