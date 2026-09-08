## Check that liveness recognizes self-XOR instructions as zeroing idioms. The
## source register is dead after each branch and can be used by the long-jump
## trampoline even without --assume-abi.

# RUN: llvm-mc -triple riscv64 -mattr=+c -filetype=obj -o %t.o %s
# RUN: ld.lld --emit-relocs -e xor_func -o %t.exe %t.o
# RUN: llvm-bolt %t.exe -o %t.xor.bolt --funcs=xor_func -split-functions \
# RUN:   -split-strategy=random2 -bolt-seed=1
# RUN: llvm-objdump -d --disassemble-symbols=xor_func %t.xor.bolt | \
# RUN:   FileCheck --check-prefix=XOR %s
# RUN: llvm-bolt %t.exe -o %t.cxor.bolt --funcs=cxor_func -split-functions \
# RUN:   -split-strategy=random2 -bolt-seed=1
# RUN: llvm-objdump -d --disassemble-symbols=cxor_func %t.cxor.bolt | \
# RUN:   FileCheck --check-prefix=CXOR %s

# XOR-LABEL: <xor_func>:
# XOR: auipc t0,
# CXOR-LABEL: <cxor_func>:
# CXOR: auipc s1,

  .text
  .option norvc
  .globl xor_func
  .type xor_func, @function
xor_func:
  beq a0, zero, .Lxor_cold
  xor s0, t0, t0
  li t0, 0
  ret
.Lxor_cold:
  xor s0, t0, t0
  li t0, 0
  ret
  .size xor_func, .-xor_func

  .option rvc
  .globl cxor_func
  .type cxor_func, @function
cxor_func:
  beq a0, zero, .Lcxor_cold
  c.xor s1, s1
  li s1, 0
  ret
.Lcxor_cold:
  c.xor s1, s1
  li s1, 0
  ret
  .size cxor_func, .-cxor_func

  .reloc 0, R_RISCV_NONE
