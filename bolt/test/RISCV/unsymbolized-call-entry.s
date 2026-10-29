// Test recovery of an RV64 linker-resolved call using the alternate link
// register and targeting an entry point inside a function.

// RUN: llvm-mc -triple riscv64 -mattr=-relax -filetype=obj -o %t.o %s
// RUN: ld.lld --no-relax --emit-relocs -o %t %t.o
// RUN: llvm-objdump -dr %t | FileCheck --check-prefix=INPUT %s
// RUN: echo target > %t.order
// RUN: echo relocated_call >> %t.order
// RUN: echo _start >> %t.order
// RUN: llvm-bolt --print-cfg --print-fix-riscv-calls --print-only=_start \
// RUN:     --reorder-functions=user \
// RUN:     --function-order=%t.order \
// RUN:     -o %t.bolt %t | FileCheck --check-prefix=BOLT %s
// RUN: llvm-objdump -d %t.bolt | FileCheck --check-prefix=OBJDUMP %s

// INPUT-LABEL: <_start>:
// INPUT:       auipc t0, 0x200
// INPUT-NEXT:  jalr t0, 0x8c(t0) <target_entry>

// BOLT-LABEL: Binary Function "_start" after building cfg {
// BOLT:       auipc t0, {{.*}}target_entry{{.*}}
// BOLT-NEXT:  jalr t0, {{.*}}(t0)
// BOLT-LABEL: Binary Function "_start" after fix-riscv-calls {
// BOLT:       call t0, {{.*}}target_entry{{.*}}

// OBJDUMP-LABEL: <target>:
// OBJDUMP:       addi a0, a0, {{(0x)?1}}
// OBJDUMP-LABEL: <target_entry>:
// OBJDUMP:       ret
// OBJDUMP-LABEL: <_start>:
// OBJDUMP:       jal t0, {{.*}} <target_entry>

  .text
  .option norvc
  .option norelax

  .globl _start
  .type _start,@function
_start:
  auipc t0, 0x200
  jalr t0, 0x8c(t0)
  ret
  .size _start, .-_start

  .skip (1 << 21) + 0x7c

  .globl target
  .type target,@function
target:
  addi a0, a0, 1
target_entry:
  ret
  .size target, .-target

  .globl relocated_call
  .type relocated_call,@function
relocated_call:
  call target
  ret
  .size relocated_call, .-relocated_call
