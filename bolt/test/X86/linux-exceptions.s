# REQUIRES: system-linux

## Check that BOLT correctly parses the Linux kernel exception table.

# RUN: llvm-mc -filetype=obj -triple x86_64-unknown-unknown %s -o %t.o
# RUN: %clang %cflags -nostdlib %t.o -o %t.exe \
# RUN:   -Wl,--image-base=0xffffffff80000000,--no-dynamic-linker,--no-eh-frame-hdr

## Verify exception bindings to instructions.

# RUN: llvm-bolt %t.exe --print-normalized -o %t.out --keep-nops=0 \
# RUN:   --bolt-info=0 | FileCheck %s

## Verify the bindings again on the rewritten binary with nops removed.

# RUN: llvm-bolt %t.out -o %t.out.1 --print-normalized | FileCheck %s

# CHECK:      BOLT-INFO: Linux kernel binary detected
# CHECK:      BOLT-INFO: parsed 2 exception table entries

  .text
  .globl _start
  .type _start, %function
_start:
# CHECK: Binary Function "_start"
  nop
.L0:
  mov (%rdi), %rax
# CHECK:      mov
# CHECK-SAME: ExceptionEntry: 1 # Fixup: [[FIXUP:[a-zA-Z0-9_]+]]
  nop
.L1:
  mov (%rsi), %rax
# CHECK:      mov
# CHECK-SAME: ExceptionEntry: 2 # Fixup: [[FIXUP]]
  nop
  ret
.LF0:
# CHECK: Secondary Entry Point: [[FIXUP]]
  jmp foo
  .size _start, .-_start

  .globl foo
  .type foo, %function
foo:
  ret
  .size foo, .-foo


## Exception table.
  .section __ex_table,"a",@progbits
  .align 4

  .long .L0 - .  # instruction
  .long .LF0 - . # fixup
  .long 0        # data

  .long .L1 - .  # instruction
  .long .LF0 - . # fixup
  .long 0        # data

## Linux kernel version
  .rodata
  .align 16
  .globl linux_banner
  .type  linux_banner, @object
linux_banner:
  .string  "Linux version 6.6.61\n"
  .size  linux_banner, . - linux_banner

## Fake Linux Kernel sections.
  .section __ksymtab,"a",@progbits
  .section __ksymtab_gpl,"a",@progbits
