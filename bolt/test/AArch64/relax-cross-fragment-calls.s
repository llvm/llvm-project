## Check call relaxation with function fragment clusters. This uses split
## functions with calls from hot and cold fragments so call thunks are placed
## around the clusters.
##
## Input layout:
##
##   A  8MiB island, pad_hot_0 50MiB, B 8MiB island, pad_hot_1 50MiB,
##   C  8MiB island, pad_hot_2 50MiB, D 8MiB island, pad_hot_3 50MiB
##
## With --split-functions, BOLT places the cold blocks of A/B/C/D in
## .text.cold. With the default 120MiB function-fragment cluster size, call
## relaxation sees:
##
##   normal layout:
##     cluster 0, .text:      A, pad_hot_0, B, pad_hot_1
##     cluster 1, .text:      C, pad_hot_2, D, pad_hot_3
##     cluster 2, .text.cold: A.cold, B.cold, C.cold, D.cold
##
##   --hot-functions-at-end:
##     cluster 0, .text.cold: A.cold, B.cold, C.cold, D.cold
##     cluster 1, .text:      A, pad_hot_0, B, pad_hot_1
##     cluster 2, .text:      C, pad_hot_2, D, pad_hot_3

# REQUIRES: system-linux

# RUN: %clang %cflags -Wl,-q -Wl,-e,A %s -o %t -nostdlib
# RUN: link_fdata --no-lbr %s %t %t.fdata
# RUN: llvm-strip --strip-unneeded %t
# RUN: llvm-bolt %t -o %t.bolt --data %t.fdata --split-functions \
# RUN:   --compact-code-model --relax-exp \
# RUN:   | FileCheck %s --check-prefix=CHECK-BOLT
# RUN: llvm-bolt %t -o %t.hfe.bolt --data %t.fdata --split-functions \
# RUN:   --compact-code-model --relax-exp --hot-functions-at-end \
# RUN:   | FileCheck %s --check-prefix=CHECK-BOLT-HFE
# RUN: llvm-readelf -S %t.bolt | FileCheck %s --check-prefix=CHECK-SECTIONS
# RUN: llvm-objdump -d \
# RUN:   --disassemble-symbols=A,B,C,D,A.cold.0,B.cold.0,C.cold.0,D.cold.0,__AArch64Thunk_A,__AArch64Thunk_C,__AArch64ADRPThunk_A \
# RUN:   %t.bolt | FileCheck %s --check-prefix=CHECK-OUTPUT
# RUN: llvm-objdump -d \
# RUN:   --disassemble-symbols=A,B,C,D,A.cold.0,B.cold.0,C.cold.0,D.cold.0,__AArch64Thunk_A,__AArch64Thunk_C,__AArch64ADRPThunk_C \
# RUN:   %t.hfe.bolt | FileCheck %s --check-prefix=CHECK-HFE-OUTPUT

# CHECK-BOLT: BOLT-INFO: built 3 function fragment cluster(s)
# CHECK-BOLT: BOLT-INFO: relaxed 4 calls with thunks
# CHECK-BOLT: BOLT-INFO: 3 short thunks created
# CHECK-BOLT: BOLT-INFO: 1 long thunks created
# CHECK-BOLT: BOLT-INFO: relaxed 8 cross-cluster branches
# CHECK-BOLT: BOLT-INFO: 12 branch thunks created

# CHECK-BOLT-HFE: BOLT-INFO: built 3 function fragment cluster(s)
# CHECK-BOLT-HFE: BOLT-INFO: relaxed 4 calls with thunks
# CHECK-BOLT-HFE: BOLT-INFO: 3 short thunks created
# CHECK-BOLT-HFE: BOLT-INFO: 1 long thunks created
# CHECK-BOLT-HFE: BOLT-INFO: relaxed 8 cross-cluster branches
# CHECK-BOLT-HFE: BOLT-INFO: 12 branch thunks created

# CHECK-SECTIONS: .text
# CHECK-SECTIONS: .text.cold

  .text
  .globl A
  .type A, %function
A:
.A_entry:
# FDATA: 1 A #.A_entry# 100
  bl B
  bl C
  cbz x0, .A_ret
  b .A_cold
.A_ret:
# FDATA: 1 A #.A_ret# 100
  ret
.A_cold:
  mov x0, #1
  bl C
  bl A
  b .A_ret
  .space 0x800000
  .size A, .-A

  .globl pad_hot_0
  .type pad_hot_0, %function
pad_hot_0:
.pad_hot_0_entry:
# FDATA: 1 pad_hot_0 #.pad_hot_0_entry# 100
  ret
  .space 0x3200000
  .size pad_hot_0, .-pad_hot_0

  .globl B
  .type B, %function
B:
.B_entry:
# FDATA: 1 B #.B_entry# 100
  cbz x0, .B_ret
  b .B_cold
.B_ret:
# FDATA: 1 B #.B_ret# 100
  ret
.B_cold:
  mov x0, #2
  b .B_ret
  .space 0x800000
  .size B, .-B

  .globl pad_hot_1
  .type pad_hot_1, %function
pad_hot_1:
.pad_hot_1_entry:
# FDATA: 1 pad_hot_1 #.pad_hot_1_entry# 100
  ret
  .space 0x3200000
  .size pad_hot_1, .-pad_hot_1

  .globl C
  .type C, %function
C:
.C_entry:
# FDATA: 1 C #.C_entry# 100
  bl A
  cbz x0, .C_ret
  b .C_cold
.C_ret:
# FDATA: 1 C #.C_ret# 100
  ret
.C_cold:
  mov x0, #3
  b .C_ret
  .space 0x800000
  .size C, .-C

  .globl pad_hot_2
  .type pad_hot_2, %function
pad_hot_2:
.pad_hot_2_entry:
# FDATA: 1 pad_hot_2 #.pad_hot_2_entry# 100
  ret
  .space 0x3200000
  .size pad_hot_2, .-pad_hot_2

  .globl D
  .type D, %function
D:
.D_entry:
# FDATA: 1 D #.D_entry# 100
  cbz x0, .D_ret
  b .D_cold
.D_ret:
# FDATA: 1 D #.D_ret# 100
  ret
.D_cold:
  mov x0, #4
  b .D_ret
  .space 0x800000
  .size D, .-D

  .globl pad_hot_3
  .type pad_hot_3, %function
pad_hot_3:
.pad_hot_3_entry:
# FDATA: 1 pad_hot_3 #.pad_hot_3_entry# 100
  ret
  .space 0x3200000
  .size pad_hot_3, .-pad_hot_3

## Force relocation mode.
  .reloc 0, R_AARCH64_NONE

# CHECK-OUTPUT:      <A>:
# CHECK-OUTPUT-NEXT: {{.*}} bl {{.*}} <B>
# CHECK-OUTPUT-NEXT: {{.*}} bl {{.*}} <__AArch64Thunk_C>

# CHECK-OUTPUT:      <__AArch64Thunk_C>:
# CHECK-OUTPUT-NEXT: {{.*}} b {{.*}} <C>

# CHECK-OUTPUT:      <__AArch64Thunk_A>:
# CHECK-OUTPUT-NEXT: {{.*}} b {{.*}} <A>

# CHECK-OUTPUT:      <C>:
# CHECK-OUTPUT-NEXT: {{.*}} bl {{.*}} <__AArch64Thunk_A>

# CHECK-OUTPUT:      <__AArch64ADRPThunk_A>:
# CHECK-OUTPUT-NEXT: {{.*}} adrp x16, {{.*}} <A>
# CHECK-OUTPUT-NEXT: {{.*}} add x16, x16, #0x0
# CHECK-OUTPUT-NEXT: {{.*}} br x16

# CHECK-OUTPUT:      <A.cold.0>:
# CHECK-OUTPUT-NEXT: {{.*}} mov x0, #0x1
# CHECK-OUTPUT-NEXT: {{.*}} bl {{.*}} <__AArch64Thunk_C>
# CHECK-OUTPUT-NEXT: {{.*}} bl {{.*}} <__AArch64ADRPThunk_A>

# CHECK-HFE-OUTPUT:      <A.cold.0>:
# CHECK-HFE-OUTPUT-NEXT: {{.*}} mov x0, #0x1
# CHECK-HFE-OUTPUT-NEXT: {{.*}} bl {{.*}} <__AArch64ADRPThunk_C>
# CHECK-HFE-OUTPUT-NEXT: {{.*}} bl {{.*}} <__AArch64Thunk_A>

# CHECK-HFE-OUTPUT:      <__AArch64ADRPThunk_C>:
# CHECK-HFE-OUTPUT-NEXT: {{.*}} adrp x16, {{.*}}
# CHECK-HFE-OUTPUT-NEXT: {{.*}} add x16, x16, #0x{{[0-9a-f]+}}
# CHECK-HFE-OUTPUT-NEXT: {{.*}} br x16

# CHECK-HFE-OUTPUT:      <__AArch64Thunk_A>:
# CHECK-HFE-OUTPUT-NEXT: {{.*}} b {{.*}} <A>

# CHECK-HFE-OUTPUT:      <A>:
# CHECK-HFE-OUTPUT-NEXT: {{.*}} bl {{.*}} <B>
# CHECK-HFE-OUTPUT-NEXT: {{.*}} bl {{.*}} <__AArch64Thunk_C>

# CHECK-HFE-OUTPUT:      <__AArch64Thunk_C>:
# CHECK-HFE-OUTPUT-NEXT: {{.*}} b {{.*}} <C>

# CHECK-HFE-OUTPUT:      <__AArch64Thunk_A>:
# CHECK-HFE-OUTPUT-NEXT: {{.*}} b {{.*}} <A>

# CHECK-HFE-OUTPUT:      <C>:
# CHECK-HFE-OUTPUT-NEXT: {{.*}} bl {{.*}} <__AArch64Thunk_A>
