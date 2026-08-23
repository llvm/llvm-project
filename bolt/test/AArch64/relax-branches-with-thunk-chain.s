## Check branch thunk chains with adjacent fragment clusters. Split functions
## have small duplicated constant islands, while hot-only pad functions create
## most of the hot-section distance.
##
## Input layout:
##
##   A  8MiB island, pad_hot_0 50MiB, B 8MiB island, pad_hot_1 50MiB,
##   C  8MiB island, pad_hot_2 50MiB, D 8MiB island, pad_hot_3 50MiB
##
## With --split-functions, BOLT places the cold blocks of A/B/C/D in
## .text.cold. With the default 120MiB function-fragment cluster size, branch
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
# RUN:   --disassemble-symbols=A,B,C,D,A.cold.0,B.cold.0,C.cold.0,D.cold.0,__AArch64BranchForwardThunk_0,__AArch64BranchForwardThunk_1,__AArch64BranchForwardThunk_4,__AArch64BranchForwardThunk_5,__AArch64BranchForwardThunk_8,__AArch64BranchForwardThunk_10,__AArch64BranchBackwardThunk_2,__AArch64BranchBackwardThunk_3,__AArch64BranchBackwardThunk_6,__AArch64BranchBackwardThunk_7,__AArch64BranchBackwardThunk_9,__AArch64BranchBackwardThunk_11 \
# RUN:   %t.bolt | FileCheck %s --check-prefix=CHECK-OUTPUT
# RUN: llvm-objdump -d \
# RUN:   --disassemble-symbols=A,B,C,D,A.cold.0,B.cold.0,C.cold.0,D.cold.0,__AArch64BranchForwardThunk_1,__AArch64BranchForwardThunk_3,__AArch64BranchForwardThunk_6,__AArch64BranchForwardThunk_7,__AArch64BranchForwardThunk_10,__AArch64BranchForwardThunk_11,__AArch64BranchBackwardThunk_0,__AArch64BranchBackwardThunk_2,__AArch64BranchBackwardThunk_4,__AArch64BranchBackwardThunk_5,__AArch64BranchBackwardThunk_8,__AArch64BranchBackwardThunk_9 \
# RUN:   %t.hfe.bolt | FileCheck %s --check-prefix=CHECK-HFE-OUTPUT

# CHECK-BOLT: BOLT-INFO: built 3 function fragment cluster(s)
# CHECK-BOLT: BOLT-INFO: relaxed 8 cross-cluster branches
# CHECK-BOLT: BOLT-INFO: 12 branch thunks created

# CHECK-BOLT-HFE: BOLT-INFO: built 3 function fragment cluster(s)
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
  cbz x0, .A_ret
  b .A_cold
.A_ret:
# FDATA: 1 A #.A_ret# 100
  ret
.A_cold:
  mov x0, #1
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

# CHECK-OUTPUT: Disassembly of section .text:

# CHECK-OUTPUT:      <A>:
# CHECK-OUTPUT-NEXT:                      {{.*}} cbnz x0, 0x[[A_BR:[0-9a-f]+]] <{{.*}}>
# CHECK-OUTPUT-NEXT: [[A_RET:[0-9a-f]+]]: {{.*}} ret
# CHECK-OUTPUT-NEXT: [[A_BR]]:            {{.*}} b    0x[[A_FW0:[0-9a-f]+]] <__AArch64BranchForwardThunk_1>

# CHECK-OUTPUT:      <B>:
# CHECK-OUTPUT-NEXT:                      {{.*}} cbnz x0, 0x[[B_BR:[0-9a-f]+]] <{{.*}}>
# CHECK-OUTPUT-NEXT: [[B_RET:[0-9a-f]+]]: {{.*}} ret
# CHECK-OUTPUT-NEXT: [[B_BR]]:            {{.*}} b    0x[[B_FW0:[0-9a-f]+]] <__AArch64BranchForwardThunk_5>

# CHECK-OUTPUT:      <__AArch64BranchForwardThunk_1>:
# CHECK-OUTPUT-NEXT: [[A_FW0]]:           {{.*}} b    0x[[A_FW1:[0-9a-f]+]] <__AArch64BranchForwardThunk_0>

# CHECK-OUTPUT:      <__AArch64BranchForwardThunk_5>:
# CHECK-OUTPUT-NEXT: [[B_FW0]]:           {{.*}} b    0x[[B_FW1:[0-9a-f]+]] <__AArch64BranchForwardThunk_4>

# CHECK-OUTPUT:      <__AArch64BranchBackwardThunk_2>:
# CHECK-OUTPUT-NEXT: [[A_BW1:[0-9a-f]+]]: {{.*}} b    0x[[A_RET]] <A+0x4>

# CHECK-OUTPUT:      <__AArch64BranchBackwardThunk_6>:
# CHECK-OUTPUT-NEXT: [[B_BW1:[0-9a-f]+]]: {{.*}} b    0x[[B_RET]] <B+0x4>

# CHECK-OUTPUT:      <C>:
# CHECK-OUTPUT-NEXT:                      {{.*}} cbnz x0, 0x[[C_BR:[0-9a-f]+]] <{{.*}}>
# CHECK-OUTPUT-NEXT: [[C_RET:[0-9a-f]+]]: {{.*}} ret
# CHECK-OUTPUT-NEXT: [[C_BR]]:            {{.*}} b    0x[[C_FW0:[0-9a-f]+]] <__AArch64BranchForwardThunk_8>

# CHECK-OUTPUT:      <D>:
# CHECK-OUTPUT-NEXT:                      {{.*}} cbnz x0, 0x[[D_BR:[0-9a-f]+]] <{{.*}}>
# CHECK-OUTPUT-NEXT: [[D_RET:[0-9a-f]+]]: {{.*}} ret
# CHECK-OUTPUT-NEXT: [[D_BR]]:            {{.*}} b    0x[[D_FW0:[0-9a-f]+]] <__AArch64BranchForwardThunk_10>

# CHECK-OUTPUT:      <__AArch64BranchForwardThunk_0>:
# CHECK-OUTPUT-NEXT: [[A_FW1]]:           {{.*}} b  0x[[A_COLD:[0-9a-f]+]] <A.cold.0>

# CHECK-OUTPUT:      <__AArch64BranchForwardThunk_4>:
# CHECK-OUTPUT-NEXT: [[B_FW1]]:           {{.*}} b  0x[[B_COLD:[0-9a-f]+]] <B.cold.0>

# CHECK-OUTPUT:      <__AArch64BranchForwardThunk_8>:
# CHECK-OUTPUT-NEXT: [[C_FW0]]:           {{.*}} b  0x[[C_COLD:[0-9a-f]+]] <C.cold.0>

# CHECK-OUTPUT:      <__AArch64BranchForwardThunk_10>:
# CHECK-OUTPUT-NEXT: [[D_FW0]]:           {{.*}} b  0x[[D_COLD:[0-9a-f]+]] <D.cold.0>

# CHECK-OUTPUT: Disassembly of section .text.cold:

# CHECK-OUTPUT:      <__AArch64BranchBackwardThunk_3>:
# CHECK-OUTPUT-NEXT: [[A_BW0:[0-9a-f]+]]: {{.*}} b   0x[[A_BW1]] <__AArch64BranchBackwardThunk_2>

# CHECK-OUTPUT:      <__AArch64BranchBackwardThunk_7>:
# CHECK-OUTPUT-NEXT: [[B_BW0:[0-9a-f]+]]: {{.*}} b   0x[[B_BW1]] <__AArch64BranchBackwardThunk_6>

# CHECK-OUTPUT:      <__AArch64BranchBackwardThunk_9>:
# CHECK-OUTPUT-NEXT: [[C_BW0:[0-9a-f]+]]: {{.*}} b   0x[[C_RET]] <C+0x4>

# CHECK-OUTPUT:      <__AArch64BranchBackwardThunk_11>:
# CHECK-OUTPUT-NEXT: [[D_BW0:[0-9a-f]+]]: {{.*}} b   0x[[D_RET]] <D+0x4>

# CHECK-OUTPUT:      <A.cold.0>:
# CHECK-OUTPUT-NEXT: [[A_COLD]]: {{.*}} mov x0, #0x1
# CHECK-OUTPUT-NEXT:             {{.*}} b   0x[[A_BW0]] <__AArch64BranchBackwardThunk_3>

# CHECK-OUTPUT:      <B.cold.0>:
# CHECK-OUTPUT-NEXT: [[B_COLD]]: {{.*}} mov x0, #0x2
# CHECK-OUTPUT-NEXT:             {{.*}} b   0x[[B_BW0]] <__AArch64BranchBackwardThunk_7>

# CHECK-OUTPUT:      <C.cold.0>:
# CHECK-OUTPUT-NEXT: [[C_COLD]]: {{.*}} mov x0, #0x3
# CHECK-OUTPUT-NEXT:             {{.*}} b   0x[[C_BW0]] <__AArch64BranchBackwardThunk_9>

# CHECK-OUTPUT:      <D.cold.0>:
# CHECK-OUTPUT-NEXT: [[D_COLD]]: {{.*}} mov x0, #0x4
# CHECK-OUTPUT-NEXT:             {{.*}} b   0x[[D_BW0]] <__AArch64BranchBackwardThunk_11>


# CHECK-HFE-OUTPUT: Disassembly of section .text.cold:

# CHECK-HFE-OUTPUT:      <A.cold.0>:
# CHECK-HFE-OUTPUT-NEXT: [[HFE_A_COLD:[0-9a-f]+]]: {{.*}} mov x0, #0x1
# CHECK-HFE-OUTPUT-NEXT:                           {{.*}} b   0x[[HFE_A_FW:[0-9a-f]+]] <__AArch64BranchForwardThunk_1>

# CHECK-HFE-OUTPUT:      <B.cold.0>:
# CHECK-HFE-OUTPUT-NEXT: [[HFE_B_COLD:[0-9a-f]+]]: {{.*}} mov x0, #0x2
# CHECK-HFE-OUTPUT-NEXT:                           {{.*}} b   0x[[HFE_B_FW:[0-9a-f]+]] <__AArch64BranchForwardThunk_3>

# CHECK-HFE-OUTPUT:      <C.cold.0>:
# CHECK-HFE-OUTPUT-NEXT: [[HFE_C_COLD:[0-9a-f]+]]: {{.*}} mov x0, #0x3
# CHECK-HFE-OUTPUT-NEXT:                           {{.*}} b   0x[[HFE_C_FW0:[0-9a-f]+]] <__AArch64BranchForwardThunk_7>

# CHECK-HFE-OUTPUT:      <D.cold.0>:
# CHECK-HFE-OUTPUT-NEXT: [[HFE_D_COLD:[0-9a-f]+]]: {{.*}} mov x0, #0x4
# CHECK-HFE-OUTPUT-NEXT:                           {{.*}} b   0x[[HFE_D_FW0:[0-9a-f]+]] <__AArch64BranchForwardThunk_11>

# CHECK-HFE-OUTPUT:      <__AArch64BranchForwardThunk_1>:
# CHECK-HFE-OUTPUT-NEXT: [[HFE_A_FW]]:             {{.*}} b   0x[[HFE_A_RET:[0-9a-f]+]] <A+0x4>

# CHECK-HFE-OUTPUT:      <__AArch64BranchForwardThunk_3>:
# CHECK-HFE-OUTPUT-NEXT: [[HFE_B_FW]]:             {{.*}} b   0x[[HFE_B_RET:[0-9a-f]+]] <B+0x4>

# CHECK-HFE-OUTPUT:      <__AArch64BranchForwardThunk_7>:
# CHECK-HFE-OUTPUT-NEXT: [[HFE_C_FW0]]:            {{.*}} b   0x[[HFE_C_FW1:[0-9a-f]+]] <__AArch64BranchForwardThunk_6>

# CHECK-HFE-OUTPUT:      <__AArch64BranchForwardThunk_11>:
# CHECK-HFE-OUTPUT-NEXT: [[HFE_D_FW0]]:            {{.*}} b   0x[[HFE_D_FW1:[0-9a-f]+]] <__AArch64BranchForwardThunk_10>

# CHECK-HFE-OUTPUT: Disassembly of section .text:

# CHECK-HFE-OUTPUT:      <__AArch64BranchBackwardThunk_0>:
# CHECK-HFE-OUTPUT-NEXT:                           {{.*}} b   0x[[HFE_A_COLD]] <A.cold.0>

# CHECK-HFE-OUTPUT:      <__AArch64BranchBackwardThunk_2>:
# CHECK-HFE-OUTPUT-NEXT:                           {{.*}} b   0x[[HFE_B_COLD]] <B.cold.0>

# CHECK-HFE-OUTPUT:      <__AArch64BranchBackwardThunk_4>:
# CHECK-HFE-OUTPUT-NEXT: [[HFE_C_BW1:[0-9a-f]+]]:  {{.*}} b   0x[[HFE_C_COLD]] <C.cold.0>

# CHECK-HFE-OUTPUT:      <__AArch64BranchBackwardThunk_8>:
# CHECK-HFE-OUTPUT-NEXT: [[HFE_D_BW1:[0-9a-f]+]]:  {{.*}} b   0x[[HFE_D_COLD]] <D.cold.0>

# CHECK-HFE-OUTPUT:      <A>:
# CHECK-HFE-OUTPUT-NEXT:                           {{.*}} cbnz x0, 0x[[HFE_A_BR:[0-9a-f]+]] <{{.*}}>
# CHECK-HFE-OUTPUT-NEXT: [[HFE_A_RET]]:            {{.*}} ret
# CHECK-HFE-OUTPUT-NEXT: [[HFE_A_BR]]:             {{.*}} b    0x{{[0-9a-f]+}} <__AArch64BranchBackwardThunk_0>

# CHECK-HFE-OUTPUT:      <B>:
# CHECK-HFE-OUTPUT-NEXT:                           {{.*}} cbnz x0, 0x[[HFE_B_BR:[0-9a-f]+]] <{{.*}}>
# CHECK-HFE-OUTPUT-NEXT: [[HFE_B_RET]]:            {{.*}} ret
# CHECK-HFE-OUTPUT-NEXT: [[HFE_B_BR]]:             {{.*}} b    0x{{[0-9a-f]+}} <__AArch64BranchBackwardThunk_2>

# CHECK-HFE-OUTPUT:      <__AArch64BranchForwardThunk_6>:
# CHECK-HFE-OUTPUT-NEXT: [[HFE_C_FW1]]:            {{.*}} b    0x[[HFE_C_RET:[0-9a-f]+]] <C+0x4>

# CHECK-HFE-OUTPUT:      <__AArch64BranchForwardThunk_10>:
# CHECK-HFE-OUTPUT-NEXT: [[HFE_D_FW1]]:            {{.*}} b    0x[[HFE_D_RET:[0-9a-f]+]] <D+0x4>

# CHECK-HFE-OUTPUT:      <__AArch64BranchBackwardThunk_5>:
# CHECK-HFE-OUTPUT-NEXT: [[HFE_C_BW0:[0-9a-f]+]]:  {{.*}} b    0x[[HFE_C_BW1]] <__AArch64BranchBackwardThunk_4>

# CHECK-HFE-OUTPUT:      <__AArch64BranchBackwardThunk_9>:
# CHECK-HFE-OUTPUT-NEXT: [[HFE_D_BW0:[0-9a-f]+]]:  {{.*}} b    0x[[HFE_D_BW1]] <__AArch64BranchBackwardThunk_8>

# CHECK-HFE-OUTPUT:      <C>:
# CHECK-HFE-OUTPUT-NEXT:                           {{.*}} cbnz x0, 0x[[HFE_C_BR:[0-9a-f]+]] <{{.*}}>
# CHECK-HFE-OUTPUT-NEXT: [[HFE_C_RET]]:            {{.*}} ret
# CHECK-HFE-OUTPUT-NEXT: [[HFE_C_BR]]:             {{.*}} b    0x[[HFE_C_BW0]] <__AArch64BranchBackwardThunk_5>

# CHECK-HFE-OUTPUT:      <D>:
# CHECK-HFE-OUTPUT-NEXT:                           {{.*}} cbnz x0, 0x[[HFE_D_BR:[0-9a-f]+]] <{{.*}}>
# CHECK-HFE-OUTPUT-NEXT: [[HFE_D_RET]]:            {{.*}} ret
# CHECK-HFE-OUTPUT-NEXT: [[HFE_D_BR]]:             {{.*}} b    0x[[HFE_D_BW0]] <__AArch64BranchBackwardThunk_9>
