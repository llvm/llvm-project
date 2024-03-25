# REQUIRES: system-linux

## Check that BOLT correctly updates the Linux kernel static keys jump table.

# RUN: llvm-mc -filetype=obj -triple x86_64-unknown-unknown %s -o %t.o
# RUN: %clang %cflags -nostdlib %t.o -o %t.exe \
# RUN:   -Wl,--image-base=0xffffffff80000000,--no-dynamic-linker,--no-eh-frame-hdr

## Verify static keys jump bindings to instructions.

# RUN: llvm-bolt %t.exe --print-normalized -o %t.out --keep-nops=0 \
# RUN:   --bolt-info=0 |& FileCheck %s

## Verify the bindings again on the rewritten binary with nops removed.

# RUN: llvm-bolt %t.out -o %t.out.1 --print-normalized |& FileCheck %s

# CHECK:      BOLT-INFO: Linux kernel binary detected
# CHECK:      BOLT-INFO: parsed 2 static keys jump entries

  .text
  .globl _start
  .type _start, %function
_start:
# CHECK: Binary Function "_start"
  nop
.L0:
  jmp .L1
# CHECK:      jit
# CHECK-SAME: # ID: 1 {{.*}} # Likely: 0 # InitValue: 1
  nop
.L1:
  .nops 5
# CHECK:      jit
# CHECK-SAME: # ID: 2 {{.*}} # Likely: 1 # InitValue: 1
.L2:
  nop
  .size _start, .-_start

  .globl foo
  .type foo, %function
foo:
  ret
  .size foo, .-foo


## Static keys jump table.
  .rodata
  .globl __start___jump_table
  .type __start___jump_table, %object
__start___jump_table:

  .long .L0 - . # Jump address
  .long .L1 - . # Target address
  .quad 1       # Key address

  .long .L1 - . # Jump address
  .long .L2 - . # Target address
  .quad 0       # Key address

  .globl __stop___jump_table
  .type __stop___jump_table, %object
__stop___jump_table:

## Fake Linux Kernel sections.
  .section __ksymtab,"a",@progbits
  .section __ksymtab_gpl,"a",@progbits
