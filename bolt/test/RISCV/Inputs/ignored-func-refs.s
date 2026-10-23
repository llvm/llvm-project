  .option norvc
  .text
  .globl leaf, middle, target, source, _start, fail
  .type leaf, @function
leaf:
  addi a0, a0, 1
  jr t0
  .size leaf, .-leaf

  .type middle, @function
middle:
  mv s0, t0
  call t0, leaf
.Lhi:
  auipc a1, %pcrel_hi(leaf)
  nop
  addi a1, a1, %pcrel_lo(.Lhi)
  lui a2, %hi(leaf)
  addi a2, a2, %lo(leaf)
  bne a1, a2, fail
  jalr t0, a1, 0
.Lindirect:
  auipc a3, %pcrel_hi(leaf)
  jalr t0, %pcrel_lo(.Lindirect)(a3)
## This pair has no relocation and targets an interior entry, clearing bit 0.
.Lcall:
  auipc t0, 0
  jalr t0, 17(t0)
  jr s0
  .size middle, .-middle

  .type target, @function
target:
  addi a0, a0, 100
target_entry:
  addi a0, a0, 1
  jr t0
  .size target, .-target

## Too small for the entry trampoline. Its JAL also cannot reach far code.
  .type source, @function
source:
  jal t0, middle
  .option push
  .option rvc
  c.jr ra
  .option pop
  .size source, .-source

  .type _start, @function
_start:
  li a0, 0
  call short_source
  call source
## Exercise the old entry even when leaf moves. Its trampoline must preserve t0.
  li a5, 0x10000
  jalr t0, a5, 0
  addi a0, a0, -5
  li a7, 93
  ecall
  .size _start, .-_start

  .type fail, @function
fail:
  li a0, 1
  li a7, 93
  ecall
  .size fail, .-fail

## Put a compressed branch close enough to new code at the next page boundary.
  .org 0xfc0
  .globl short_target, short_source
  .type short_target, @function
short_target:
  ret
  .size short_target, .-short_target

  .type short_source, @function
short_source:
  .option push
  .option rvc
  .option exact
  c.beqz a0, short_target
  .option pop
  ret
  .size short_source, .-short_source

## Link separately with -u end_refs to retain this section under --gc-sections.
  .section .text.end_refs,"ax",@progbits
  .globl end_refs
  .type end_refs, @function
end_refs:
  lla a1, _end
  lui a1, %hi(_end)
  addi a1, a1, %lo(_end)
  ret
  .size end_refs, .-end_refs
