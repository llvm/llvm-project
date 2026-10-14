## Two symbols at the same address give different permissions to their branch
## relocations. The branch to the NOTYPE symbol must preserve x16; the branch
## to the FUNC symbol may use an x16-clobbering veneer.

# REQUIRES: system-linux

# RUN: %clang %cflags -Wl,-q -Wl,-e,foo %s -o %t -nostdlib
# RUN: link_fdata --no-lbr %s %t %t.fdata
# RUN: llvm-bolt %t -o %t.bolt --data %t.fdata --lite=0 --relax-exp \
# RUN:   --max-thunk-chain-length=0 --reorder-functions=exec-count \
# RUN:   | FileCheck %s --check-prefix=BOLT
# RUN: llvm-objdump -d --disassemble-symbols=foo,baz,__AArch64_forward_Thunk_0,__AArch64_forward_ADRPThunk_body_func_0 \
# RUN:   %t.bolt | FileCheck %s --check-prefix=DISASM

# BOLT: relaxed 1 calls with long thunks
# BOLT: 1 long thunks created
# BOLT: relaxed 1 unconditional branches
# BOLT: 1 branch thunks created

# DISASM:      <foo>:
# DISASM:        mov  x16, #0x2a
# DISASM-NEXT:   b    {{.*}} <__AArch64_forward_Thunk_0>
# DISASM:      <baz>:
# DISASM:        b    {{.*}} <__AArch64_forward_ADRPThunk_body_func_0>
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
  b body_notype
  .size foo, .-foo

  .globl baz
  .type baz, %function
baz:
# FDATA: 1 baz #baz# 100
  mov x0, #1
  b body_func
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
