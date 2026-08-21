// Test recovery of a linker-resolved AUIPC/JALR pair that has no relocation,
// even though the rest of the executable retains relocations.

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
// INPUT:       auipc ra, 0x200
// INPUT-NEXT:  jalr 0x88(ra) <target>
// INPUT-NEXT:  ret
// INPUT-LABEL: <relocated_call>:
// INPUT:       R_RISCV_CALL_PLT target

// BOLT-LABEL: Binary Function "_start" after building cfg {
// BOLT:       auipc ra, target
// BOLT-NEXT:  jalr {{.*}}(ra)
// BOLT-LABEL: Binary Function "_start" after fix-riscv-calls {
// BOLT:       call target

// OBJDUMP-LABEL: <target>:
// OBJDUMP-LABEL: <_start>:
// OBJDUMP:       jal {{.*}} <target>

  .text
  .option norvc
  .option norelax

  .globl _start
  .type _start,@function
_start:
  // The target starts 0x200088 bytes after this AUIPC. Spell out the resolved
  // immediates so this pair has no relocation, as happens after LTO linking.
  auipc ra, 0x200
  jalr ra, 0x88(ra)
  ret
  .size _start, .-_start

  .skip (1 << 21) + 0x7c

  .globl target
  .type target,@function
target:
  ret
  .size target, .-target

  .globl relocated_call
  .type relocated_call,@function
relocated_call:
  call target
  ret
  .size relocated_call, .-relocated_call
