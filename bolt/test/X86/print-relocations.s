
# This test is to check if -print-relocations option works correctly.

# REQUIRES: system-linux

# RUN: llvm-mc -filetype=obj -triple x86_64-unknown-unknown \
# RUN:   %s -o %t.o
# RUN: ld.lld -q %t.o -o %t.exe -q
# RUN: llvm-bolt %t.exe -o %t.bolt.exe -print-only=_start \
# RUN:   -print-disasm -print-relocations | FileCheck %s

# CHECK: leaq    foo(%rip), %rax # Relocs: (R: R_X86_64_REX_GOTPCRELX
  .globl _start
  .type _start, %function
_start:
  movq 0(%rip), %rax
  .reloc .-4, R_X86_64_REX_GOTPCRELX, foo-4
  ret

  .globl foo
  .type foo, %function
foo:
  nop
