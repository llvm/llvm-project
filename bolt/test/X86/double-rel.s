## Check that BOLT can correctly use relocations to symbolize instruction
## operands when an instruction can have up to two relocations associated
## with it.

# REQUIRES: system-linux

# RUN: llvm-mc -filetype=obj -triple x86_64-unknown-linux %s -o %t.o
# RUN: ld.lld %t.o -o %t.exe -q --Tdata=0x80000
# RUN: llvm-bolt %t.exe --relocs -o /dev/null --print-only=_start --print-disasm \
# RUN:   | FileCheck %s --check-prefix=CHECK-BOLT
# RUN: llvm-objdump -d --print-imm-hex %t.exe \
# RUN:   | FileCheck %s --check-prefix=CHECK-OBJDUMP

  .data
  .globl VAR
VAR:
  .quad

  .text
  .globl _start
  .type _start,@function
_start:
  .cfi_startproc

## VAR value is 0x80000. Using relocations, llvm-bolt should correctly
## symbolize the instruction operands.

  movq $VAR, 0x80000
# CHECK-BOLT:    movq $VAR, 0x80000
# CHECK-OBJDUMP: movq $0x80000, 0x80000

  movq $0x80000, VAR
# CHECK-BOLT-NEXT:    movq $0x80000, VAR
# CHECK-OBJDUMP-NEXT: movq $0x80000, 0x80000

  movq $VAR, VAR
# CHECK-BOLT-NEXT:    movq $VAR, VAR
# CHECK-OBJDUMP-NEXT: movq $0x80000, 0x80000

  retq
  .size _start, .-_start
  .cfi_endproc
