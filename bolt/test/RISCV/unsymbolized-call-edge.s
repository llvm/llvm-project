// Test signed target reconstruction, JALR target-bit clearing, tail calls,
// and rejection of AUIPC/JALR near matches without relocations.

// RUN: llvm-mc -triple riscv64 -mattr=-relax -filetype=obj -o %t.o %s
// RUN: ld.lld --no-relax --emit-relocs -e backward_target -o %t %t.o
// RUN: echo forward_target > %t.order
// RUN: echo odd_call >> %t.order
// RUN: echo near_matches >> %t.order
// RUN: echo backward_tail >> %t.order
// RUN: echo backward_target >> %t.order
// RUN: echo relocated_call >> %t.order
// RUN: llvm-bolt --print-cfg --print-fix-riscv-calls \
// RUN:     --print-only=backward_tail --print-only=odd_call \
// RUN:     --print-only=near_matches --reorder-functions=user \
// RUN:     --function-order=%t.order \
// RUN:     -o %t.bolt %t | FileCheck --check-prefix=BOLT %s
// RUN: llvm-objdump -d %t.bolt | FileCheck --check-prefix=OBJDUMP %s
// RUN: llvm-mc -triple riscv32 -mattr=-relax -filetype=obj -o %t.32.o %s
// RUN: ld.lld --no-relax --emit-relocs -e backward_target -o %t.32 %t.32.o
// RUN: llvm-bolt --print-cfg --print-only=odd_call -o %t.32.bolt %t.32 \
// RUN:     | FileCheck --check-prefix=RV32 %s

// BOLT-LABEL: Binary Function "backward_tail" after building cfg {
// BOLT:       auipc t1, backward_target
// BOLT-NEXT:  jr {{.*}}(t1)
// BOLT-LABEL: Binary Function "odd_call" after building cfg {
// BOLT:       auipc ra, forward_target
// BOLT-NEXT:  jalr {{.*}}(ra)
// BOLT-LABEL: Binary Function "near_matches" after building cfg {
// BOLT:       auipc t0, 0x0
// BOLT-NEXT:  jalr t1
// BOLT-NEXT:  auipc t0, 0x0
// BOLT-NEXT:  jalr t0
// BOLT-LABEL: Binary Function "backward_tail" after fix-riscv-calls {
// BOLT:       nop
// BOLT-NEXT:  tail backward_target
// BOLT-LABEL: Binary Function "odd_call" after fix-riscv-calls {
// BOLT:       nop
// BOLT-NEXT:  call forward_target
// BOLT-LABEL: Binary Function "near_matches" after fix-riscv-calls {
// BOLT:       auipc t0, 0x0
// BOLT-NEXT:  jalr t1
// BOLT-NEXT:  auipc t0, 0x0
// BOLT-NEXT:  jalr t0

// RV32-LABEL: Binary Function "odd_call" after building cfg {
// RV32:       auipc ra, 0
// RV32-NEXT:  jalr 0x89(ra)

// OBJDUMP-LABEL: <odd_call>:
// OBJDUMP:       jal {{.*}} <forward_target>
// OBJDUMP-LABEL: <near_matches>:
// OBJDUMP:       auipc t0, 0x0
// OBJDUMP-NEXT:  jalr t1
// OBJDUMP-NEXT:  auipc t0, 0x0
// OBJDUMP-NEXT:  jalr t0
// OBJDUMP-LABEL: <backward_tail>:
// OBJDUMP:       j {{.*}} <backward_target>

  .text
  .option norvc
  .option norelax

  .globl backward_target
  .type backward_target,@function
backward_target:
  ret
  .size backward_target, .-backward_target

  .skip 0x1000

  .globl backward_tail
  .type backward_tail,@function
backward_tail:
  // backward_target is 0x1004 bytes before this AUIPC.
  auipc t1, 0xfffff
  jalr zero, -4(t1)
  .size backward_tail, .-backward_tail

  .globl odd_call
  .type odd_call,@function
odd_call:
  // JALR clears bit zero, so 0x89 targets forward_target at offset 0x88.
  auipc ra, 0
  jalr ra, 0x89(ra)
  ret
  .size odd_call, .-odd_call

  .skip 0x7c

  .globl forward_target
  .type forward_target,@function
forward_target:
  ret
  .size forward_target, .-forward_target

  .globl near_matches
  .type near_matches,@function
near_matches:
  // The JALR base does not match the AUIPC destination.
  auipc t0, 0
  jalr ra, 0(t1)
  // The JALR link register is neither zero nor the AUIPC destination.
  auipc t0, 0
  jalr ra, 0(t0)
  ret
  .size near_matches, .-near_matches

  // Retain a relocation so BOLT can reorder the functions containing the
  // linker-resolved instruction pairs above.
  .globl relocated_call
  .type relocated_call,@function
relocated_call:
  call forward_target
  ret
  .size relocated_call, .-relocated_call
