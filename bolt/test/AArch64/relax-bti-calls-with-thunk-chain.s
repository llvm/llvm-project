## Check BTI fixups and call relaxation with clustered layout:
##  > A (old text) calls B (new text), whose entry needs BTI.
##  > D (new text) calls A (old text), whose entry NOP is patched to BTI.
##  > In-layout calls use short thunk chains, which need no additional BTI.

# REQUIRES: system-linux

# RUN: llvm-mc -filetype=obj -triple aarch64-unknown-unknown \
# RUN:   -mattr=+bti -aarch64-mark-bti-property %s -o %t.o
# RUN: %clang %cflags %t.o -o %t -nostdlib -Wl,-q,-z,force-bti -Wl,-e,A
# RUN: link_fdata --no-lbr %s %t %t.fdata
# RUN: llvm-strip --strip-unneeded %t
# RUN: llvm-bolt %t -o %t.bolt --data %t.fdata --split-functions \
# RUN:   --relax-exp --max-cluster-size=124 --skip-funcs=A \
# RUN:   | FileCheck %s --check-prefix=CHECK-BOLT
# RUN: llvm-objdump -d %t.bolt | FileCheck %s --check-prefix=CHECK-OUTPUT

# CHECK-BOLT: BOLT-INFO: built 3 function fragment cluster(s)
# CHECK-BOLT-NEXT: BOLT-INFO: cluster: 0
# CHECK-BOLT-NEXT: BOLT-INFO:   2 fragment(s)
# CHECK-BOLT-NEXT: BOLT-INFO:   100 estimated bytes without thunks
# CHECK-BOLT: BOLT-INFO: cluster: 1
# CHECK-BOLT-NEXT: BOLT-INFO:   2 fragment(s)
# CHECK-BOLT-NEXT: BOLT-INFO:   124 estimated bytes without thunks
# CHECK-BOLT: BOLT-INFO: cluster: 2
# CHECK-BOLT-NEXT: BOLT-INFO:   2 fragment(s)
# CHECK-BOLT-NEXT: BOLT-INFO:   120 estimated bytes without thunks
# CHECK-BOLT: BOLT-INFO: relaxed 2 calls with short thunks
# CHECK-BOLT: BOLT-INFO: relaxed 1 calls with long thunks
# CHECK-BOLT: BOLT-INFO: 4 short thunks created
# CHECK-BOLT: BOLT-INFO: 1 long thunks created
# CHECK-BOLT: BOLT-INFO: relaxed 2 unconditional branches
# CHECK-BOLT: BOLT-INFO: 4 branch thunks created

  .text
  .globl A
  .type A, %function
A:
  nop
  bl B
  ret
  .size A, .-A

  .globl B
  .type B, %function
B:
.B_entry:
# FDATA: 1 B #.B_entry# 100
  cbz x0, .L_B_cold
.B_ret:
# FDATA: 1 B #.B_ret# 100
  ret
.L_B_cold_first:
  mov x0, #2
  ret
.L_B_cold:
  mov x0, #1
  bl C
  cbz x1, .L_B_cold_first
  ret
  .space 0x10
  .size B, .-B

  .globl C
  .type C, %function
C:
.C_entry:
# FDATA: 1 C #.C_entry# 100
  cbz x0, .L_C_cold
.C_ret:
# FDATA: 1 C #.C_ret# 100
  ret
.L_C_cold:
  bl B
  ret
  .space 0x20
  .size C, .-C

  .globl D
  .type D, %function
D:
.D_entry:
# FDATA: 1 D #.D_entry# 100
  bl A
  ret
  .space 0x2c
  .size D, .-D

  .globl E
  .type E, %function
E:
.E_entry:
# FDATA: 1 E #.E_entry# 100
  ret
  .space 0x30
  .size E, .-E

## Force relocation mode.
  .reloc 0, R_AARCH64_NONE

## Keep the relocated B beyond the range of A's direct call.
  .data
  .space 0x8000000

# CHECK-OUTPUT:      Disassembly of section .bolt.org.text:

# CHECK-OUTPUT:      [[#%.16x,A_ADDR:]] <A>:
# CHECK-OUTPUT-NEXT: {{.*}} bti c
# CHECK-OUTPUT-NEXT: {{.*}} bl {{.*}} <B.org.0>
# CHECK-OUTPUT-NEXT: {{.*}} ret

# CHECK-OUTPUT:      <B.org.0>:
# CHECK-OUTPUT-NEXT: {{.*}} adrp x16, {{.*}} <B>
# CHECK-OUTPUT-NEXT: {{.*}} add x16, x16, #0x0
# CHECK-OUTPUT-NEXT: {{.*}} br x16

# CHECK-OUTPUT:      Disassembly of section .text:

########################## Cluster 0 ##########################
#
# CHECK-OUTPUT:      <B>:
# CHECK-OUTPUT-NEXT: {{.*}} bti c
# CHECK-OUTPUT-NEXT: {{.*}} cbz x0, {{.*}} <B+0xc>
# CHECK-OUTPUT-NEXT: {{.*}} ret
# CHECK-OUTPUT-NEXT: {{.*}} b {{.*}} <__AArch64_forward_Thunk_1>
#
# CHECK-OUTPUT:      <C>:
# CHECK-OUTPUT-NEXT: {{.*}} cbz x0, {{.*}} <C+0x8>
# CHECK-OUTPUT-NEXT: {{.*}} ret
# CHECK-OUTPUT-NEXT: {{.*}} b {{.*}} <__AArch64_forward_Thunk_3>
#
# CHECK-OUTPUT:      <__AArch64_forward_Thunk_1>:
# CHECK-OUTPUT-NEXT: {{.*}} b {{.*}} <__AArch64_forward_Thunk_0>
#
# CHECK-OUTPUT:      <__AArch64_forward_Thunk_3>:
# CHECK-OUTPUT-NEXT: {{.*}} b {{.*}} <__AArch64_forward_Thunk_2>

########################## Cluster 1 ##########################
#
# CHECK-OUTPUT:      <__AArch64_backward_Thunk_C_0>:
# CHECK-OUTPUT-NEXT: {{.*}} b {{.*}} <C>
#
# CHECK-OUTPUT:      <__AArch64_backward_Thunk_B_2>:
# CHECK-OUTPUT-NEXT: {{.*}} b {{.*}} <B>
#
# CHECK-OUTPUT:      <D>:
# CHECK-OUTPUT-NEXT: {{.*}} bl {{.*}} <__AArch64_forward_ADRPThunk_A_0>
# CHECK-OUTPUT-NEXT: {{.*}} ret
#
# CHECK-OUTPUT:      <E>:
# CHECK-OUTPUT-NEXT: {{.*}} ret
#
# CHECK-OUTPUT:      <__AArch64_forward_ADRPThunk_A_0>:
# CHECK-OUTPUT-NEXT: {{.*}} adrp x16, 0x[[#%x,A_PAGE:]]
# CHECK-OUTPUT-NEXT: {{.*}} add x16, x16, #0x[[#%x,A_ADDR-A_PAGE]]
# CHECK-OUTPUT-NEXT: {{.*}} br x16
#
# CHECK-OUTPUT:      <__AArch64_forward_Thunk_0>:
# CHECK-OUTPUT-NEXT: {{.*}} b {{.*}} <B.cold.0+0x8>
#
# CHECK-OUTPUT:      <__AArch64_forward_Thunk_2>:
# CHECK-OUTPUT-NEXT: {{.*}} b {{.*}} <C.cold.0>

# CHECK-OUTPUT:      Disassembly of section .text.cold:

########################## Cluster 2 ##########################
#
# CHECK-OUTPUT:      <__AArch64_backward_Thunk_C_1>:
# CHECK-OUTPUT-NEXT: {{.*}} b {{.*}} <__AArch64_backward_Thunk_C_0>
#
# CHECK-OUTPUT:      <__AArch64_backward_Thunk_B_3>:
# CHECK-OUTPUT-NEXT: {{.*}} b {{.*}} <__AArch64_backward_Thunk_B_2>
#
# CHECK-OUTPUT:      <B.cold.0>:
# CHECK-OUTPUT-NEXT: {{.*}} mov x0, #0x2
# CHECK-OUTPUT-NEXT: {{.*}} ret
# CHECK-OUTPUT-NEXT: {{.*}} mov x0, #0x1
# CHECK-OUTPUT-NEXT: {{.*}} bl {{.*}} <__AArch64_backward_Thunk_C_1>
# CHECK-OUTPUT-NEXT: {{.*}} cbz x1, {{.*}} <B.cold.0>
# CHECK-OUTPUT-NEXT: {{.*}} ret
#
# CHECK-OUTPUT:      <C.cold.0>:
# CHECK-OUTPUT-NEXT: {{.*}} bl {{.*}} <__AArch64_backward_Thunk_B_3>
# CHECK-OUTPUT-NEXT: {{.*}} ret
