// Check that BOLT materializes an explicit tail call for functions connected
// by an implicit fall-through so they can be moved independently.

// RUN: split-file %s %t.dir
// RUN: llvm-mc -triple=riscv64 -mattr=+c -filetype=obj \
// RUN:   -o %t.dir/input.o %t.dir/input.s
// RUN: ld.lld -q -e _start -o %t.dir/input.exe %t.dir/input.o
// RUN: llvm-bolt %t.dir/input.exe -o %t.dir/output.exe \
// RUN:   --reorder-functions=user --function-order=%t.dir/order.txt 2>&1 \
// RUN:   | FileCheck %s --check-prefix=WARNING
// RUN: llvm-objdump -d --no-show-raw-insn %t.dir/output.exe \
// RUN:   | FileCheck %s --check-prefix=DISASM
// RUN: llvm-nm -n %t.dir/output.exe | FileCheck %s --check-prefix=NM

// WARNING-NOT: interprocedural fall-through detected from terminated_source
// WARNING: BOLT-WARNING: interprocedural fall-through detected from fallthrough_source to fallthrough_target; materializing an explicit tail call
// WARNING-NOT: interprocedural fall-through detected from terminated_source

// DISASM-LABEL: <fallthrough_source>:
// DISASM:       li a0, 0x0
// DISASM-NEXT:  jal {{.*}} <fallthrough_target>

// NM:      T fallthrough_source
// NM-NEXT: T poison
// NM-NEXT: T fallthrough_target

//--- input.s
  .text
  .option rvc

  .globl _start
  .type _start, @function
  .p2align 2
_start:
  call fallthrough_source
  li a7, 93
  ecall
  j .
  .size _start, .-_start

  .globl fallthrough_source
  .type fallthrough_source, @function
  .p2align 2
fallthrough_source:
  li a0, 0
  .size fallthrough_source, .-fallthrough_source

  // Keep this alignment padding outside fallthrough_source's symbol size.
  // Execution intentionally crosses it to reach fallthrough_target.
  .p2align 2
  .globl fallthrough_target
  .type fallthrough_target, @function
fallthrough_target:
  ret
  .size fallthrough_target, .-fallthrough_target

  .globl poison
  .type poison, @function
  .p2align 2
poison:
  li a0, 1
  ret
  .size poison, .-poison

  // The padding after this terminator must not be mistaken for a fall-through.
  .globl terminated_source
  .type terminated_source, @function
  .p2align 2
terminated_source:
  ret
  .size terminated_source, .-terminated_source

  .p2align 2
  .globl terminated_target
  .type terminated_target, @function
terminated_target:
  ret
  .size terminated_target, .-terminated_target

//--- order.txt
_start
fallthrough_source
poison
fallthrough_target
terminated_source
terminated_target
