## Relax direct calls using thunks across function fragment clusters. A/B/C/D
## contain 56MiB islands. With --split-functions, BOLT places the cold blocks
## of A/B/C/D in .text.cold. The test exercises forward and backward short
## thunks for adjacent clusters, as well as long thunk reuse across remote
## clusters. With the default 124MiB function-fragment cluster size, call
## relaxation sees:
##
##   A -> B             same cluster, direct
##   B -> C             forward short thunk
##   C -> A             backward short thunk
##   D -> A             shares backward short thunk to A
##
##   B.cold -> B        normal: backward long thunk
##   C.cold -> B        normal: shares backward long thunk to B
##   D.cold -> B        normal: shares backward long thunk to B
##
##   B.cold -> D        HFE: forward long thunk
##   D.cold -> D        HFE: shares forward long thunk to D
##
##   normal layout:
##     cluster 0, .text:      A, B
##     cluster 1, .text:      C, D
##     cluster 2, .text.cold: A.cold, B.cold
##     cluster 3, .text.cold: C.cold, D.cold
##
##   --hot-functions-at-end:
##     cluster 0: .text.cold A.cold, B.cold
##     cluster 1: .text.cold C.cold, D.cold
##     cluster 2: .text A, B
##     cluster 3: .text C, D

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
# RUN:   --disassemble-symbols=A,B,C,D,A.cold.0,B.cold.0,C.cold.0,D.cold.0,__AArch64Thunk_A,__AArch64Thunk_C,__AArch64Thunk_D,__AArch64ADRPThunk_B,__AArch64ADRPThunk_D \
# RUN:   %t.bolt | FileCheck %s --check-prefix=CHECK-OUTPUT
# RUN: llvm-objdump -d \
# RUN:   --disassemble-symbols=A,B,C,D,A.cold.0,B.cold.0,C.cold.0,D.cold.0,__AArch64Thunk_A,__AArch64Thunk_B,__AArch64Thunk_C,__AArch64ADRPThunk_B,__AArch64ADRPThunk_D \
# RUN:   %t.hfe.bolt | FileCheck %s --check-prefix=CHECK-HFE-OUTPUT

# CHECK-BOLT: BOLT-INFO: built 4 function fragment cluster(s)
# CHECK-BOLT: BOLT-INFO: relaxed 4 adjacent cluster calls with thunks
# CHECK-BOLT: BOLT-INFO: relaxed 4 remote cluster calls with thunks
# CHECK-BOLT: BOLT-INFO: 3 short thunks created
# CHECK-BOLT: BOLT-INFO: 2 long thunks created
# CHECK-BOLT: BOLT-INFO: 1 short thunks reused
# CHECK-BOLT: BOLT-INFO: 2 long thunks reused
# CHECK-BOLT: BOLT-INFO: relaxed 8 cross-cluster branches
# CHECK-BOLT: BOLT-INFO: 16 branch thunks created

# CHECK-BOLT-HFE: BOLT-INFO: built 4 function fragment cluster(s)
# CHECK-BOLT-HFE: BOLT-INFO: relaxed 5 adjacent cluster calls with thunks
# CHECK-BOLT-HFE: BOLT-INFO: relaxed 3 remote cluster calls with thunks
# CHECK-BOLT-HFE: BOLT-INFO: 3 short thunks created
# CHECK-BOLT-HFE: BOLT-INFO: 2 long thunks created
# CHECK-BOLT-HFE: BOLT-INFO: 2 short thunks reused
# CHECK-BOLT-HFE: BOLT-INFO: 1 long thunks reused
# CHECK-BOLT-HFE: BOLT-INFO: relaxed 8 cross-cluster branches
# CHECK-BOLT-HFE: BOLT-INFO: 16 branch thunks created

# CHECK-SECTIONS: .text
# CHECK-SECTIONS: .text.cold

  .text
  .globl A
  .type A, %function
A:
.A_entry:
# FDATA: 1 A #.A_entry# 100
  bl B
  cbz x0, .A_ret
  b .A_cold
.A_ret:
# FDATA: 1 A #.A_ret# 100
  ret
.A_cold:
  mov x0, #1
  b .A_ret
  .space 0x3800000
  .size A, .-A

  .globl B
  .type B, %function
B:
.B_entry:
# FDATA: 1 B #.B_entry# 100
  bl C
  cbz x0, .B_ret
  b .B_cold
.B_ret:
# FDATA: 1 B #.B_ret# 100
  ret
.B_cold:
  mov x0, #2
  bl B
  bl D
  b .B_ret
  .space 0x3800000
  .size B, .-B

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
  bl B
  b .C_ret
  .space 0x3800000
  .size C, .-C

  .globl D
  .type D, %function
D:
.D_entry:
# FDATA: 1 D #.D_entry# 100
  bl A
  cbz x0, .D_ret
  b .D_cold
.D_ret:
# FDATA: 1 D #.D_ret# 100
  ret
.D_cold:
  mov x0, #4
  bl B
  bl D
  b .D_ret
  .space 0x3800000
  .size D, .-D

## Force relocation mode.
  .reloc 0, R_AARCH64_NONE

# CHECK-OUTPUT:      <A>:
# CHECK-OUTPUT-NEXT: {{.*}} bl {{.*}} <B>

# CHECK-OUTPUT:      <B>:
# CHECK-OUTPUT-NEXT: {{.*}} bl {{.*}} <__AArch64Thunk_C>

# CHECK-OUTPUT:      <__AArch64Thunk_C>:
# CHECK-OUTPUT-NEXT: {{.*}} b {{.*}} <C>

# CHECK-OUTPUT:      <__AArch64Thunk_A>:
# CHECK-OUTPUT-NEXT: {{.*}} b {{.*}} <A>

# CHECK-OUTPUT:      <C>:
# CHECK-OUTPUT-NEXT: {{.*}} bl {{.*}} <__AArch64Thunk_A>

# CHECK-OUTPUT:      <D>:
# CHECK-OUTPUT-NEXT: {{.*}} bl {{.*}} <__AArch64Thunk_A>

# CHECK-OUTPUT:      <__AArch64Thunk_D>:
# CHECK-OUTPUT-NEXT: {{.*}} b {{.*}} <D>

# CHECK-OUTPUT:      <A.cold.0>:

# CHECK-OUTPUT:      <B.cold.0>:
# CHECK-OUTPUT-NEXT: {{.*}} mov x0, #0x2
# CHECK-OUTPUT-NEXT: {{.*}} bl {{.*}} <__AArch64ADRPThunk_B>
# CHECK-OUTPUT-NEXT: {{.*}} bl {{.*}} <__AArch64Thunk_D>

# CHECK-OUTPUT:      <__AArch64ADRPThunk_B>:
# CHECK-OUTPUT-NEXT: {{.*}} adrp x16, {{.*}}
# CHECK-OUTPUT-NEXT: {{.*}} add x16, x16, {{.*}}
# CHECK-OUTPUT-NEXT: {{.*}} br x16

# CHECK-OUTPUT:      <__AArch64ADRPThunk_D>:
# CHECK-OUTPUT-NEXT: {{.*}} adrp x16, {{.*}}
# CHECK-OUTPUT-NEXT: {{.*}} add x16, x16, {{.*}}
# CHECK-OUTPUT-NEXT: {{.*}} br x16

# CHECK-OUTPUT:      <C.cold.0>:
# CHECK-OUTPUT-NEXT: {{.*}} mov x0, #0x3
# CHECK-OUTPUT-NEXT: {{.*}} bl {{.*}} <__AArch64ADRPThunk_B>

# CHECK-OUTPUT:      <D.cold.0>:
# CHECK-OUTPUT-NEXT: {{.*}} mov x0, #0x4
# CHECK-OUTPUT-NEXT: {{.*}} bl {{.*}} <__AArch64ADRPThunk_B>
# CHECK-OUTPUT-NEXT: {{.*}} bl {{.*}} <__AArch64ADRPThunk_D>

# CHECK-HFE-OUTPUT:      <A.cold.0>:

# CHECK-HFE-OUTPUT:      <B.cold.0>:
# CHECK-HFE-OUTPUT-NEXT: {{.*}} mov x0, #0x2
# CHECK-HFE-OUTPUT-NEXT: {{.*}} bl {{.*}} <__AArch64ADRPThunk_B>
# CHECK-HFE-OUTPUT-NEXT: {{.*}} bl {{.*}} <__AArch64ADRPThunk_D>

# CHECK-HFE-OUTPUT:      <__AArch64ADRPThunk_D>:
# CHECK-HFE-OUTPUT-NEXT: {{.*}} adrp x16, {{.*}}
# CHECK-HFE-OUTPUT-NEXT: {{.*}} add x16, x16, {{.*}}
# CHECK-HFE-OUTPUT-NEXT: {{.*}} br x16

# CHECK-HFE-OUTPUT:      <__AArch64ADRPThunk_B>:
# CHECK-HFE-OUTPUT-NEXT: {{.*}} adrp x16, {{.*}}
# CHECK-HFE-OUTPUT-NEXT: {{.*}} add x16, x16, {{.*}}
# CHECK-HFE-OUTPUT-NEXT: {{.*}} br x16

# CHECK-HFE-OUTPUT:      <C.cold.0>:
# CHECK-HFE-OUTPUT-NEXT: {{.*}} mov x0, #0x3
# CHECK-HFE-OUTPUT-NEXT: {{.*}} bl {{.*}} <__AArch64Thunk_B>

# CHECK-HFE-OUTPUT:      <D.cold.0>:
# CHECK-HFE-OUTPUT-NEXT: {{.*}} mov x0, #0x4
# CHECK-HFE-OUTPUT-NEXT: {{.*}} bl {{.*}} <__AArch64Thunk_B>
# CHECK-HFE-OUTPUT-NEXT: {{.*}} bl {{.*}} <__AArch64ADRPThunk_D>

# CHECK-HFE-OUTPUT:      <__AArch64Thunk_B>:
# CHECK-HFE-OUTPUT-NEXT: {{.*}} b {{.*}} <B>

# CHECK-HFE-OUTPUT:      <A>:
# CHECK-HFE-OUTPUT-NEXT: {{.*}} bl {{.*}} <B>

# CHECK-HFE-OUTPUT:      <B>:
# CHECK-HFE-OUTPUT-NEXT: {{.*}} bl {{.*}} <__AArch64Thunk_C>

# CHECK-HFE-OUTPUT:      <__AArch64Thunk_C>:
# CHECK-HFE-OUTPUT-NEXT: {{.*}} b {{.*}} <C>

# CHECK-HFE-OUTPUT:      <__AArch64Thunk_A>:
# CHECK-HFE-OUTPUT-NEXT: {{.*}} b {{.*}} <A>

# CHECK-HFE-OUTPUT:      <C>:
# CHECK-HFE-OUTPUT-NEXT: {{.*}} bl {{.*}} <__AArch64Thunk_A>

# CHECK-HFE-OUTPUT:      <D>:
# CHECK-HFE-OUTPUT-NEXT: {{.*}} bl {{.*}} <__AArch64Thunk_A>
