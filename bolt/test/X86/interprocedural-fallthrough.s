# REQUIRES: x86_64-linux

## Check that BOLT materializes an explicit tail call for functions connected
## by an implicit fall-through so they can be moved independently.

# RUN: split-file %s %t.dir
# RUN: llvm-mc -filetype=obj -triple x86_64-unknown-linux-gnu \
# RUN:   %t.dir/input.s -o %t.dir/input.o
# RUN: %clang %cflags -no-pie %t.dir/input.o -o %t.dir/input.exe \
# RUN:   -Wl,--no-dynamic-linker -Wl,-q -Wl,-e,_start
# RUN: %t.dir/input.exe
# RUN: llvm-bolt %t.dir/input.exe -o %t.dir/output.exe \
# RUN:   --reorder-functions=user --function-order=%t.dir/order.txt 2>&1 \
# RUN:   | FileCheck %s --check-prefix=WARNING
# RUN: %t.dir/output.exe
# RUN: llvm-nm -n %t.dir/output.exe | FileCheck %s --check-prefix=NM
# RUN: llvm-objdump -d --no-show-raw-insn \
# RUN:   --disassemble-symbols=fallthrough_source %t.dir/output.exe \
# RUN:   | FileCheck %s --check-prefix=DISASM

# WARNING-NOT: interprocedural fall-through detected from terminated_source
# WARNING: BOLT-WARNING: interprocedural fall-through detected from fallthrough_source to fallthrough_target; materializing an explicit tail call
# WARNING-NOT: interprocedural fall-through detected from terminated_source

# NM:      T fallthrough_source
# NM-NEXT: T poison
# NM-NEXT: T fallthrough_target
# DISASM:  jmp {{.*}} <fallthrough_target>

#--- input.s
  .text

  .globl _start
  .type _start, @function
  .p2align 4
_start:
  callq fallthrough_source
  movq $60, %rax
  syscall
  ud2
  .size _start, .-_start

  .globl fallthrough_source
  .type fallthrough_source, @function
  .p2align 4
fallthrough_source:
  xorl %edi, %edi
  .size fallthrough_source, .-fallthrough_source

  # Keep this alignment padding outside fallthrough_source's symbol size.
  # Execution intentionally crosses it to reach fallthrough_target.
  .p2align 4
  .globl fallthrough_target
  .type fallthrough_target, @function
fallthrough_target:
  retq
  .size fallthrough_target, .-fallthrough_target

  .globl poison
  .type poison, @function
  .p2align 4
poison:
  movl $1, %edi
  retq
  .size poison, .-poison

  # The padding after this terminator must not be mistaken for a fall-through.
  .globl terminated_source
  .type terminated_source, @function
  .p2align 4
terminated_source:
  retq
  .size terminated_source, .-terminated_source

  .p2align 4
  .globl terminated_target
  .type terminated_target, @function
terminated_target:
  retq
  .size terminated_target, .-terminated_target

#--- order.txt
_start
fallthrough_source
poison
fallthrough_target
terminated_source
terminated_target
