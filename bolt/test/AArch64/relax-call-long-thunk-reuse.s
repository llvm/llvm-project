## Check that long call thunk reuse only uses thunks emitted by the adjacent
## cluster. Borrowed thunks must not be cached as if they were emitted by the
## borrowing cluster, otherwise reuse can become transitive and pick a thunk at
## the wrong cluster boundary.
##
## With --max-cluster-size=64, each function forms its own cluster:
##
##   cluster 0: A
##   cluster 1: B
##   cluster 2: C -> A
##   cluster 3: D -> A
##   cluster 4: E -> A
##
## E creates a backward long thunk to A in cluster 4. D can reuse it from the
## adjacent cluster boundary, but C needs a separate thunk in cluster 2.

# REQUIRES: system-linux

# RUN: %clang %cflags -Wl,-q -Wl,-e,A %s -o %t -nostdlib
# RUN: link_fdata --no-lbr %s %t %t.fdata
# RUN: llvm-strip --strip-unneeded %t
# RUN: llvm-bolt %t -o %t.bolt --data %t.fdata \
# RUN:   --compact-code-model --relax-exp --max-cluster-size=64 \
# RUN:   | FileCheck %s --check-prefix=CHECK-BOLT
# RUN: llvm-objdump -d \
# RUN:   --disassemble-symbols=A,B,C,D,E,__AArch64_backward_long_call_A_0,__AArch64_backward_long_call_A_1 \
# RUN:   %t.bolt | FileCheck %s --check-prefix=CHECK-OUTPUT

# CHECK-BOLT: BOLT-INFO: built 5 function fragment cluster(s)
# CHECK-BOLT: BOLT-INFO: relaxed 3 long cluster calls with thunks
# CHECK-BOLT: BOLT-INFO: 2 long thunks created
# CHECK-BOLT: BOLT-INFO: 1 long thunks reused

  .text
  .globl A
  .type A, %function
A:
.A_entry:
# FDATA: 1 A #.A_entry# 100
  ret
  .space 0x30
  .size A, .-A

  .globl B
  .type B, %function
B:
.B_entry:
# FDATA: 1 B #.B_entry# 100
  ret
  .space 0x30
  .size B, .-B

  .globl C
  .type C, %function
C:
.C_entry:
# FDATA: 1 C #.C_entry# 100
  bl A
  ret
  .space 0x30
  .size C, .-C

  .globl D
  .type D, %function
D:
.D_entry:
# FDATA: 1 D #.D_entry# 100
  bl A
  ret
  .space 0x30
  .size D, .-D

  .globl E
  .type E, %function
E:
.E_entry:
# FDATA: 1 E #.E_entry# 100
  bl A
  ret
  .space 0x30
  .size E, .-E

## Force relocation mode.
  .reloc 0, R_AARCH64_NONE

# CHECK-OUTPUT:      <A>:
# CHECK-OUTPUT-NEXT: {{.*}} ret

# CHECK-OUTPUT:      <B>:
# CHECK-OUTPUT-NEXT: {{.*}} ret

# CHECK-OUTPUT:      <__AArch64_backward_long_call_A_1>:
# CHECK-OUTPUT-NEXT: {{.*}} adrp x16, {{.*}}
# CHECK-OUTPUT-NEXT: {{.*}} add x16, x16, {{.*}}
# CHECK-OUTPUT-NEXT: {{.*}} br x16

# CHECK-OUTPUT:      <C>:
# CHECK-OUTPUT-NEXT: {{.*}} bl {{.*}} <__AArch64_backward_long_call_A_1>

# CHECK-OUTPUT:      <D>:
# CHECK-OUTPUT-NEXT: {{.*}} bl {{.*}} <__AArch64_backward_long_call_A_0>

# CHECK-OUTPUT:      <__AArch64_backward_long_call_A_0>:
# CHECK-OUTPUT-NEXT: {{.*}} adrp x16, {{.*}}
# CHECK-OUTPUT-NEXT: {{.*}} add x16, x16, {{.*}}
# CHECK-OUTPUT-NEXT: {{.*}} br x16

# CHECK-OUTPUT:      <E>:
# CHECK-OUTPUT-NEXT: {{.*}} bl {{.*}} <__AArch64_backward_long_call_A_0>
