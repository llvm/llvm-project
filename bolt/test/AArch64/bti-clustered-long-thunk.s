## Clustered long thunks use br x16 to reach a rewritten function entry and
## an internal basic block. Both targets need BTI landing pads.

# REQUIRES: system-linux

# RUN: llvm-mc -filetype=obj -triple aarch64-unknown-unknown \
# RUN:   -mattr=+bti -aarch64-mark-bti-property %s -o %t.o
# RUN: %clang %cflags -Wl,-q -Wl,-e,A %t.o -o %t.exe -nostdlib -Wl,-z,force-bti
# RUN: link_fdata --no-lbr %s %t.exe %t.fdata
# RUN: llvm-bolt %t.exe -o %t.bolt --data %t.fdata \
# RUN:   --relax-exp --max-cluster-size=64 \
# RUN:   --max-thunk-chain-length=0 | FileCheck %s --check-prefix=CHECK-BOLT
# RUN: llvm-objdump -d -j .text %t.bolt \
# RUN:   | FileCheck %s --check-prefix=CHECK-OUTPUT

# CHECK-BOLT: BOLT-INFO: binary is using BTI
# CHECK-BOLT: BOLT-INFO: 2 long thunks created

# CHECK-OUTPUT-LABEL: <A>:
# CHECK-OUTPUT-NEXT: bti c
# CHECK-OUTPUT-NEXT: cbz x0, {{0x[0-9a-f]+}} <.A_internal>
# CHECK-OUTPUT-NEXT: ret

# CHECK-OUTPUT-LABEL: <.A_internal>:
# CHECK-OUTPUT-NEXT: bti c
# CHECK-OUTPUT-NEXT: ret
# CHECK-OUTPUT-NEXT: nop

# CHECK-OUTPUT-LABEL: <B>:
# CHECK-OUTPUT-NEXT: ret
# CHECK-OUTPUT-NEXT: nop

# CHECK-OUTPUT-LABEL: <__AArch64_backward_ADRPThunk_.A_internal>:
# CHECK-OUTPUT-NEXT: adrp x16, {{0x[0-9a-f]+}} <A>
# CHECK-OUTPUT-NEXT: add x16, x16, #{{0x[0-9a-f]+}}
# CHECK-OUTPUT-NEXT: br x16

# CHECK-OUTPUT-LABEL: <C>:
# CHECK-OUTPUT-NEXT: bl {{0x[0-9a-f]+}} <__AArch64_backward_ADRPThunk_.A_internal>
# CHECK-OUTPUT-NEXT: ret
# CHECK-OUTPUT-NEXT: nop

# CHECK-OUTPUT-LABEL: <__AArch64_backward_ADRPThunk_A_0>:
# CHECK-OUTPUT-NEXT: adrp x16, {{0x[0-9a-f]+}} <A>
# CHECK-OUTPUT-NEXT: add x16, x16, #{{0x[0-9a-f]+}}
# CHECK-OUTPUT-NEXT: br x16

# CHECK-OUTPUT-LABEL: <D>:
# CHECK-OUTPUT-NEXT: bl {{0x[0-9a-f]+}} <__AArch64_backward_ADRPThunk_A_0>
# CHECK-OUTPUT-NEXT: ret
# CHECK-OUTPUT-NEXT: nop

  .text
  .globl A
  .type A, %function
A:
.A_entry:
# FDATA: 1 A #.A_entry# 100
  cbz x0, .A_internal
  ret
.A_internal:
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
  bl .A_internal
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

  .reloc 0, R_AARCH64_NONE
