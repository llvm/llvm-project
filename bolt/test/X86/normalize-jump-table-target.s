# This test reproduces an issue with removing a basic block which is referenced
# from rodata section ("externally referenced offset"/jump table target).
# If the block is removed (by remove-nops + NormalizeCFG), and BOLT updates the
# original jump table (with jump-tables=move), this causes an emission error
# (undefined temporary symbol).

# REQUIRES: system-linux,bolt-runtime

# RUN: llvm-mc -filetype=obj -triple x86_64-unknown-unknown %s -o %t.o
# RUN: %clang -no-pie %t.o -o %t.exe -Wl,-q
# RUN: llvm-bolt %t.exe -o %t.out --instrument --instrumentation-file=%t.tmp \
# RUN:   2>&1 | FileCheck %s --check-prefix=CHECK-BOLT

# CHECK-BOLT-NOT:  undefined temporary symbol

# Grab the jump table address in original text section (.bolt.org.text)
# RUN: llvm-objdump -dj.bolt.org.text %t.out > %t.log
# Inspect jump table entries, should be two zero values
# RUN: llvm-objdump -sj.rodata %t.out >> %t.log
# RUN: FileCheck %s --input-file %t.log --check-prefix=CHECK-OBJDUMP

# CHECK-OBJDUMP: jmpq *0x[[#%x,JTADDR:]](,%rax,8)
# CHECK-OBJDUMP: [[#JTADDR]] 00000000 00000000 00000000 00000000

  .globl main
main:
  .cfi_startproc
  jmp main
  jmpq  *JT(,%rax,8)
a:
  shlq  %rax
.Ltmp:
  nop
c:
  ret
  .cfi_endproc

.rodata
# pad to ensure the jump table starts at 16-byte aligned address so it can be
# matched in objdump output
.p2align 4
JT:
  .quad .Ltmp
  .quad a
