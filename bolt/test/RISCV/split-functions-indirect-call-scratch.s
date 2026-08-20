## Check that register scavenging for a cross-fragment long jump accounts for
## the explicit target register of an indirect call in the destination block.
## The trampoline must not clobber t0 before the cold block calls through it.

# RUN: llvm-mc -triple riscv64 -mattr=+c -filetype=obj -o %t.o %s
# RUN: ld.lld --emit-relocs -e _start -o %t.exe %t.o
# RUN: llvm-bolt %t.exe -o %t.bolt -split-functions \
# RUN:   -split-strategy=random2 -bolt-seed=1
# RUN: llvm-objdump -d --no-show-raw-insn %t.bolt | FileCheck %s

# CHECK-LABEL: <_start>:
# CHECK:       auipc t0,
# CHECK-NEXT:  addi t0, t0,
# CHECK:       auipc t1,
# CHECK-NEXT:  {{(jalr zero,|jr)}} {{.*}}(t1)
# CHECK:       auipc t1,
# CHECK-NEXT:  {{(jalr zero,|jr)}} {{.*}}(t1)
# CHECK-LABEL: <_start.cold.0>:
# CHECK:       jalr t0

  .text
  .globl _start
  .type _start, @function
_start:
1:
  auipc t0, %pcrel_hi(callee)
  addi t0, t0, %pcrel_lo(1b)
  beq a0, zero, .Lcold
  li a0, 1
  ret
.Lcold:
  jalr ra, t0, 0
  ret
  .size _start, .-_start

  .globl callee
  .type callee, @function
callee:
  li a0, 0
  ret
  .size callee, .-callee

  .reloc 0, R_RISCV_NONE
