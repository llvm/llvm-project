# REQUIRES: riscv
## The regression test led to oscillation between two states ("address assignment did not converge" error).
## First jump (~2^11 bytes away): alternated between 4 and 8 bytes.
## Second jump (~2^20 bytes away): alternated between 2 and 8 bytes.

# RUN: rm -rf %t && split-file %s %t && cd %t
# RUN: llvm-mc -filetype=obj -triple=riscv64 -mattr=+c,+relax a.s -o a.o
# RUN: ld.lld -T lds a.o -o out
# RUN: llvm-objdump -td --no-show-raw-insn -M no-aliases out | FileCheck %s

# CHECK-LABEL: <_start>:
# CHECK-NEXT:                  jal     zero, {{.*}} <low>
# CHECK-EMPTY:
# CHECK-NEXT:  <jump_high>:
# CHECK-NEXT:      1004:       auipc   t0, 0x100
# CHECK-NEXT:                  jalr    zero, -0x2(t0) <high>
# CHECK-NEXT:                  ...
# CHECK:       <low>:
# CHECK:           1802:       c.jr    ra

# CHECK:       <high>:
# CHECK-NEXT:    101002:       jal     ra, 0x1004 <jump_high>
# CHECK-NEXT:                  auipc   ra, 0xfff00
# CHECK-NEXT:                  jalr    ra, -0x2(ra) <jump_high>

#--- a.s
## At the beginning of state1, low-_start = 0x800-2, reachable by a c.j.
## This increases high-jump_high from 0x100000-2 to 0x100000, unreachable by a jal.
## In the next iteration, low-_start increases to 0x800, unreachable by a c.j.
.global _start
_start:
  jump low, t0    # state0: 4 bytes; state1: 2 bytes
jump_high:
  jump high, t0   # state0: 4 bytes; state1: 8 bytes

  .space 0x800-10
low:
  ret

.section .high,"ax",@progbits
  nop
high:
  call jump_high
  call jump_high

#--- lds
SECTIONS {
  .text 0x1000 : { *(.text) }
  .high 0x101000 : { *(.high) }
}
