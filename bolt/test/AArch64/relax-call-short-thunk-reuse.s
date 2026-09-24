## Check that short call thunks are reused. With --max-cluster-size=160,
## A/B form the source cluster, C/D/E separate the target cluster, and F is the
## call target. Without large alignment, measured thunk islands collapse the
## conservative reservations and all calls remain direct. With 32MiB text
## alignment, A and B share one thunk while C and D remain in direct Branch26
## range of F.
##
##   cluster 0: A -> F, B -> F
##   cluster 1: C -> F
##   cluster 2: D -> F
##   cluster 3: E
##   cluster 4: F

# REQUIRES: system-linux

# RUN: %clang %cflags -Wl,-q -Wl,-e,A %s -o %t -nostdlib
# RUN: link_fdata --no-lbr %s %t %t.fdata
# RUN: llvm-strip --strip-unneeded %t
# RUN: llvm-bolt %t -o %t.conservative --data %t.fdata \
# RUN:   --compact-code-model --relax-exp --max-cluster-size=160 \
# RUN:   --max-thunk-chain-length=3 --max-thunk-remeasure=0 \
# RUN:   | FileCheck %s --check-prefix=CHECK-CONSERVATIVE
# RUN: llvm-bolt %t -o %t.measured --data %t.fdata \
# RUN:   --compact-code-model --relax-exp --max-cluster-size=160 \
# RUN:   --max-thunk-chain-length=3 \
# RUN:   | FileCheck %s --check-prefix=CHECK-MEASURED \
# RUN:   --implicit-check-not='BOLT-INFO: relaxed'
# RUN: llvm-bolt %t -o %t.fixed-point --data %t.fdata \
# RUN:   --compact-code-model --relax-exp --max-cluster-size=160 \
# RUN:   --max-thunk-chain-length=3 --max-thunk-remeasure=3 \
# RUN:   | FileCheck %s --check-prefix=CHECK-FIXED-POINT \
# RUN:   --implicit-check-not='BOLT-INFO: relaxed'
# RUN: llvm-bolt %t -o %t.bolt --data %t.fdata \
# RUN:   --compact-code-model --relax-exp --max-cluster-size=160 \
# RUN:   --max-thunk-chain-length=3 --align-text=33554432 \
# RUN:   | FileCheck %s --check-prefix=CHECK-BOLT
# RUN: not llvm-bolt %t -o %t.invalid --data %t.fdata \
# RUN:   --compact-code-model --relax-exp --max-cluster-size=134217727 \
# RUN:   2>&1 | FileCheck %s --check-prefix=CHECK-INVALID
# RUN: llvm-objdump -d \
# RUN:   --disassemble-symbols=A,B,C,D,E,F,__AArch64_forward_Thunk_F_0 \
# RUN:   %t.bolt | FileCheck %s --check-prefix=CHECK-OUTPUT

# CHECK-CONSERVATIVE: BOLT-INFO: built 5 function fragment cluster(s)
# CHECK-CONSERVATIVE: BOLT-INFO: thunk island layout remeasurement disabled
# CHECK-CONSERVATIVE: BOLT-INFO: relaxed 3 calls with short thunks
# CHECK-CONSERVATIVE: BOLT-INFO: 2 short thunks created
# CHECK-CONSERVATIVE: BOLT-INFO: 1 short thunks reused

# CHECK-MEASURED: BOLT-INFO: built 5 function fragment cluster(s)
# CHECK-MEASURED: BOLT-INFO: thunk island layout did not stabilize after 1 remeasurement iteration

# CHECK-FIXED-POINT: BOLT-INFO: built 5 function fragment cluster(s)
# CHECK-FIXED-POINT: BOLT-INFO: thunk island layout stabilized after 3 remeasurement iterations

# CHECK-BOLT: BOLT-INFO: built 5 function fragment cluster(s)
# CHECK-BOLT: BOLT-INFO: relaxed 2 calls with short thunks
# CHECK-BOLT: BOLT-INFO: 1 short thunks created
# CHECK-BOLT: BOLT-INFO: 1 short thunks reused

# CHECK-INVALID: BOLT-ERROR: --max-cluster-size leaves only 0 bytes per thunk island, less than the required layout alignment of 2097152 bytes

## B reuses the thunk created for A, accounting for the single reuse.

  .text
  .globl A
  .type A, %function
A:
.A_entry:
# FDATA: 1 A #.A_entry# 100
  bl F
  ret
  .space 0x30
  .size A, .-A

  .globl B
  .type B, %function
B:
.B_entry:
# FDATA: 1 B #.B_entry# 100
  bl F
  ret
  .space 0x30
  .size B, .-B

  .globl C
  .type C, %function
C:
.C_entry:
# FDATA: 1 C #.C_entry# 100
  bl F
  ret
  .space 0x60
  .size C, .-C

  .globl D
  .type D, %function
D:
.D_entry:
# FDATA: 1 D #.D_entry# 100
  bl F
  ret
  .space 0x60
  .size D, .-D

  .globl E
  .type E, %function
E:
.E_entry:
# FDATA: 1 E #.E_entry# 100
  ret
  .space 0x80
  .size E, .-E

  .globl F
  .type F, %function
F:
.F_entry:
# FDATA: 1 F #.F_entry# 100
  ret
  .space 0x30
  .size F, .-F

## Force relocation mode.
  .reloc 0, R_AARCH64_NONE

# CHECK-OUTPUT:      <A>:
# CHECK-OUTPUT-NEXT: {{.*}} bl {{.*}} <__AArch64_forward_Thunk_F_0>
# CHECK-OUTPUT-NEXT: {{.*}} ret

# CHECK-OUTPUT:      <B>:
# CHECK-OUTPUT-NEXT: {{.*}} bl {{.*}} <__AArch64_forward_Thunk_F_0>
# CHECK-OUTPUT-NEXT: {{.*}} ret

# CHECK-OUTPUT:      <C>:
# CHECK-OUTPUT-NEXT: {{.*}} bl {{.*}} <F>
# CHECK-OUTPUT-NEXT: {{.*}} ret

# CHECK-OUTPUT:      <D>:
# CHECK-OUTPUT-NEXT: {{.*}} bl {{.*}} <F>
# CHECK-OUTPUT-NEXT: {{.*}} ret

# CHECK-OUTPUT:      <E>:
# CHECK-OUTPUT-NEXT: {{.*}} ret

# CHECK-OUTPUT:      <__AArch64_forward_Thunk_F_0>:
# CHECK-OUTPUT-NEXT: {{.*}} b {{.*}} <F>

# CHECK-OUTPUT:      <F>:
# CHECK-OUTPUT-NEXT: {{.*}} ret
