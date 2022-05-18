# This reproduces a bug with misinterpreting the gotpcrelx reloc

# Here we use llvm-mc -relax-relocations to produce R_X86_64_REX_GOTPCRELX
# and ld.lld to consume it and optimize it, transforming a CMP <mem, reg>
# into CMP <imm, reg>.
# Then we check that BOLT updates correctly the imm operand that references
# a function address. Currently XFAIL as we do not support it.

# REQUIRES: system-linux
# XFAIL: *

# RUN: llvm-mc -filetype=obj -triple x86_64-unknown-linux \
# RUN:   -relax-relocations %s -o %t.o
# RUN: llvm-strip --strip-unneeded %t.o
# RUN: ld.lld %t.o -o %t.exe -q
# RUN: llvm-readobj -r %t.exe | FileCheck --check-prefix=READOBJ %s
# RUN: llvm-bolt %t.exe -relocs -o %t.out -lite=0
# RUN: llvm-objdump -d --no-show-raw-insn --print-imm-hex \
# RUN:   %t.out | FileCheck --check-prefix=DISASM %s

# Check that R_X86_64_REX_GOTPCRELX is present in the input binary
# READOBJ: 0x[[#%X,]] R_X86_64_REX_GOTPCRELX foo 0x[[#%X,]]

# DISASM:      Disassembly of section .text:
# DISASM-EMPTY:
# DISASM-NEXT: <_start>:
# DISASM-NEXT:                 leaq  0x[[#%x,ADDR:]], %rax
# DISASM-NEXT:                 cmpq  0x[[#ADDR]], %rax

  .text
  .globl _start
  .type _start, %function
_start:
  .cfi_startproc
  leaq foo, %rax
  cmpq foo@GOTPCREL(%rip), %rax
  je  b
c:
  mov $1, %rdi
  callq foo
b:
  xorq %rdi, %rdi
  callq foo
  ret
  .cfi_endproc
  .size _start, .-_start

  .globl foo
  .type foo, %function
foo:
  .cfi_startproc
  ret
  .cfi_endproc
  .size foo, .-foo
