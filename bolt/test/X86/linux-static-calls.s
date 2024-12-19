# REQUIRES: system-linux

## Check that BOLT correctly updates the Linux kernel static calls table.

# RUN: llvm-mc -filetype=obj -triple x86_64-unknown-unknown %s -o %t.o
# RUN: %clang %cflags -nostdlib %t.o -o %t.exe \
# RUN:   -Wl,--image-base=0xffffffff80000000,--no-dynamic-linker,--no-eh-frame-hdr

## Verify static calls bindings to instructions.

# RUN: llvm-bolt %t.exe --print-normalized -o %t.out --keep-nops=0 \
# RUN:   --bolt-info=0 2>&1 | FileCheck %s

## Verify the bindings again on the rewritten binary with nops removed.

# RUN: llvm-bolt %t.out -o %t.out.1 --print-normalized 2>&1 | FileCheck %s

# CHECK:      BOLT-INFO: Linux kernel binary detected
# CHECK:      BOLT-INFO: parsed 2 static call entries

  .text
  .globl _start
  .type _start, %function
_start:
# CHECK: Binary Function "_start"
  nop
.L0:
  call foo
# CHECK:      callq foo           # {{.*}} StaticCall: 1
  nop
.L1:
  jmp foo
# CHECK:      jmp foo             # {{.*}} StaticCall: 2
  .size _start, .-_start

  .globl foo
  .type foo, %function
foo:
  ret
  .size foo, .-foo


## Static call table.
  .rodata
  .globl __start_static_call_sites
  .type __start_static_call_sites, %object
__start_static_call_sites:
  .long .L0 - .
  .long 0
  .long .L1 - .
  .long 0

  .globl __stop_static_call_sites
  .type __stop_static_call_sites, %object
__stop_static_call_sites:

## Fake Linux Kernel sections.
  .section __ksymtab,"a",@progbits
  .section __ksymtab_gpl,"a",@progbits
