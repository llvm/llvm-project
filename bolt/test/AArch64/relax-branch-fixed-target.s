## Check B-only relaxation to a fixed target outside the clustered output.
## In the first run, a thunk at the leading output boundary can bridge the
## branch. In the second, the entire rewritten output is over 128 MiB from the
## fixed target, so no B-only thunk chain can reach it.

# REQUIRES: system-linux

# RUN: %clang %cflags -Wl,-q -Wl,-e,anchor %s -o %t -nostdlib
# RUN: link_fdata --no-lbr %s %t %t.fdata
# RUN: llvm-strip --strip-unneeded %t
# RUN: llvm-bolt %t -o %t.bolt --data %t.fdata --relocs --lite=0 \
# RUN:   --relax-exp --reorder-functions=exec-count --skip-funcs=target \
# RUN:   --pad-funcs=anchor:73400320 --pad-funcs=separator:62914560 \
# RUN:   | FileCheck %s --check-prefix=CHECK-BOLT
# RUN: llvm-objdump -d \
# RUN:   --disassemble-symbols=target,target_body,anchor,source,__AArch64_backward_Thunk_0 \
# RUN:   %t.bolt | FileCheck %s --check-prefix=CHECK-OUTPUT
# RUN: not llvm-bolt %t -o %t.unreachable --data %t.fdata --relocs --lite=0 \
# RUN:   --relax-exp --reorder-functions=exec-count --skip-funcs=target \
# RUN:   --pad-funcs-before=anchor:134217728 2>&1 \
# RUN:   | FileCheck %s --check-prefix=CHECK-UNREACHABLE

# CHECK-BOLT: BOLT-INFO: relaxed 1 unconditional branches
# CHECK-BOLT: BOLT-INFO: 1 branch thunks created

# CHECK-OUTPUT:      <target_body>:
# CHECK-OUTPUT-NEXT: {{.*}} cmp x16, #0x2a

# CHECK-OUTPUT:      <__AArch64_backward_Thunk_0>:
# CHECK-OUTPUT-NEXT: {{.*}} b {{.*}} <target_body>

# CHECK-OUTPUT:      <source>:
# CHECK-OUTPUT-NEXT: {{.*}} mov x16, #0x2a
# CHECK-OUTPUT-NEXT: {{.*}} b {{.*}} <__AArch64_backward_Thunk_0>

# CHECK-UNREACHABLE: BOLT-ERROR: unable to build branch thunk chain
# CHECK-UNREACHABLE-SAME: to target_body
# CHECK-UNREACHABLE-SAME: outside the clustered layout

  .text
  .globl target
  .type target, %function
target:
  b .Ltarget_ret
  .globl target_body
  .type target_body, %notype
target_body:
  cmp x16, #42
  cset w0, ne
.Ltarget_ret:
  ret
  .size target, .-target

  .globl anchor
  .type anchor, %function
anchor:
.anchor_entry:
# FDATA: 1 anchor #.anchor_entry# 100
  ret
  .size anchor, .-anchor

  .globl separator
  .type separator, %function
separator:
.separator_entry:
# FDATA: 1 separator #.separator_entry# 75
  ret
  .size separator, .-separator

  .globl source
  .type source, %function
source:
.source_entry:
# FDATA: 1 source #.source_entry# 50
  mov x16, #42
  b target_body
  .size source, .-source

## Force relocation mode.
  .reloc 0, R_AARCH64_NONE
