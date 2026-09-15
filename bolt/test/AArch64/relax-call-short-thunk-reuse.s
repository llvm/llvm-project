## Check that short call thunks are reused. With --max-cluster-size=160,
## A/B form the source cluster, C/D/E separate the target cluster, and F is the
## call target. A builds a three-hop thunk chain. B reuses the whole chain,
## while C and D reuse the rungs already placed after their clusters.
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
# RUN: llvm-bolt %t -o %t.bolt --data %t.fdata \
# RUN:   --compact-code-model --relax-exp --max-cluster-size=160 \
# RUN:   --max-thunk-chain-length=3 \
# RUN:   | FileCheck %s --check-prefix=CHECK-BOLT
# RUN: llvm-objdump -d \
# RUN:   --disassemble-symbols=A,B,C,D,E,F,__AArch64_forward_Thunk_F_0,__AArch64_forward_Thunk_F_1,__AArch64_forward_Thunk_F_2 \
# RUN:   %t.bolt | FileCheck %s --check-prefix=CHECK-OUTPUT

# CHECK-BOLT: BOLT-INFO: built 5 function fragment cluster(s)
# CHECK-BOLT: BOLT-INFO: relaxed 4 calls with short thunks
# CHECK-BOLT: BOLT-INFO: 3 short thunks created
# CHECK-BOLT: BOLT-INFO: 6 short thunks reused

## The reuse counter is per reused chain rung:
##
##   B -> F  reuses cluster 0, cluster 1, cluster 2 thunks  = 3
##   C -> F  reuses cluster 1, cluster 2 thunks             = 2
##   D -> F  reuses cluster 2 thunk                         = 1
##
##   Total reused thunk lookups:
##
##   3 + 2 + 1 = 6

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
# CHECK-OUTPUT-NEXT: {{.*}} bl {{.*}} <__AArch64_forward_Thunk_F_2>
# CHECK-OUTPUT-NEXT: {{.*}} ret

# CHECK-OUTPUT:      <B>:
# CHECK-OUTPUT-NEXT: {{.*}} bl {{.*}} <__AArch64_forward_Thunk_F_2>
# CHECK-OUTPUT-NEXT: {{.*}} ret

# CHECK-OUTPUT:      <__AArch64_forward_Thunk_F_2>:
# CHECK-OUTPUT-NEXT: {{.*}} b {{.*}} <__AArch64_forward_Thunk_F_1>

# CHECK-OUTPUT:      <C>:
# CHECK-OUTPUT-NEXT: {{.*}} bl {{.*}} <__AArch64_forward_Thunk_F_1>
# CHECK-OUTPUT-NEXT: {{.*}} ret

# CHECK-OUTPUT:      <__AArch64_forward_Thunk_F_1>:
# CHECK-OUTPUT-NEXT: {{.*}} b {{.*}} <__AArch64_forward_Thunk_F_0>

# CHECK-OUTPUT:      <D>:
# CHECK-OUTPUT-NEXT: {{.*}} bl {{.*}} <__AArch64_forward_Thunk_F_0>
# CHECK-OUTPUT-NEXT: {{.*}} ret

# CHECK-OUTPUT:      <__AArch64_forward_Thunk_F_0>:
# CHECK-OUTPUT-NEXT: {{.*}} b {{.*}} <F>

# CHECK-OUTPUT:      <E>:
# CHECK-OUTPUT-NEXT: {{.*}} ret

# CHECK-OUTPUT:      <F>:
# CHECK-OUTPUT-NEXT: {{.*}} ret
