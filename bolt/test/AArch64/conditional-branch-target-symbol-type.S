## Conditional branches to a NOTYPE symbol in another function can carry live
## x16. They must reach the target through B-only branch thunks. A FUNC alias
## at the same address can still use a long call thunk.

# REQUIRES: system-linux

# RUN: %clang %cflags -Wl,-q -Wl,-e,foo %s -o %t -nostdlib
# RUN: link_fdata --no-lbr %s %t %t.fdata
# RUN: llvm-bolt %t -o %t.bolt --data %t.fdata --lite=0 --relax-exp \
# RUN:   --max-thunk-chain-length=0 --reorder-functions=exec-count \
# RUN:   | FileCheck %s --check-prefix=BOLT
# RUN: llvm-objdump -d --disassemble-symbols=foo,baz,__AArch64_forward_Thunk_0,__AArch64_forward_ADRPThunk_body_func_0 \
# RUN:   %t.bolt | FileCheck %s --check-prefix=DISASM
# RUN: link_fdata %s %t %t.lbr.fdata FDATA-LBR
# RUN: llvm-bolt %t -o %t.lbr.bolt --data %t.lbr.fdata -w %t.yaml \
# RUN:   --lite=0 --relax-exp --print-cfg | FileCheck %s --check-prefix=PROFILE
# RUN: llvm-bolt %t -o %t.yaml.bolt --data %t.yaml \
# RUN:   --lite=0 --relax-exp --print-cfg | FileCheck %s --check-prefix=PROFILE

# FDATA-LBR: 0 [unknown] 0 1 foo 0 0 100
# FDATA-LBR: 1 foo 8 1 body_notype 0 2 60
# FDATA-LBR: 1 foo c 1 body_notype 0 3 30

# PROFILE-LABEL: Binary Function "foo" after building cfg {
# PROFILE:      Exec Count  : 100
# PROFILE:      cbz x2, {{.*}} # Count: 60 # MispredCount: 2
# PROFILE:      Successors: {{.*}} (mispreds: 2, count: 60)
# PROFILE:      tbz w3, #0x0, {{.*}} # Count: 30 # MispredCount: 3
# PROFILE:      Successors: {{.*}} (mispreds: 3, count: 30)

# BOLT: relaxed 1 calls with long thunks
# BOLT: relaxed 2 unconditional branches
# BOLT: 1 branch thunks created

# DISASM:      <foo>:
# DISASM:        mov x16, #0x2a
# DISASM:        cbz x2,       0x[[EB1:[0-9a-f]+]] <foo+{{.*}}>
# DISASM-NEXT:   tbz w3, #0x0, 0x[[EB2:[0-9a-f]+]] <foo+{{.*}}>
# DISASM-NEXT:   ret
# DISASM-NEXT: [[EB1]]: {{.*}} b   {{.*}} <__AArch64_forward_Thunk_0>
# DISASM-NEXT: [[EB2]]: {{.*}} b   {{.*}} <__AArch64_forward_Thunk_0>
# DISASM:      <baz>:
# DISASM:        cbz x2, {{.*}} <baz+{{.*}}>
# DISASM:        b       {{.*}} <__AArch64_forward_ADRPThunk_body_func_0>
# DISASM:      <__AArch64_forward_ADRPThunk_body_func_0>:
# DISASM-NEXT:   adrp x16,
# DISASM-NEXT:   add  x16, x16,
# DISASM-NEXT:   br   x16
# DISASM:      <__AArch64_forward_Thunk_0>:
# DISASM-NEXT:   b    {{.*}} <body_func>

  .text
  .globl bar
  .type bar, %function
bar:
  b .bar_ret
  .globl body_notype
  .type body_notype, %notype
body_notype:
  .globl body_func
  .type body_func, %function
body_func:
  cbnz x0, .bar_ret
  cmp x16, #42
  cset w0, ne
  ret
.bar_ret:
  ret
  .size bar, .-bar

  .globl foo
  .type foo, %function
foo:
# FDATA: 1 foo #foo# 100
  mov x0, #0
  mov x16, #42
  cbz x2, body_notype
  tbz x3, #0, body_notype
  ret
  .size foo, .-foo

  .globl baz
  .type baz, %function
baz:
# FDATA: 1 baz #baz# 100
  mov x0, #1
  cbz x2, body_func
  ret
  .size baz, .-baz

  .globl padding1
  .type padding1, %function
padding1:
# FDATA: 1 padding1 #padding1# 100
  ret
  .space 0x4400000
  .size padding1, .-padding1

  .globl padding2
  .type padding2, %function
padding2:
# FDATA: 1 padding2 #padding2# 100
  ret
  .space 0x4400000
  .size padding2, .-padding2

## Force relocation mode.
  .reloc 0, R_AARCH64_NONE
