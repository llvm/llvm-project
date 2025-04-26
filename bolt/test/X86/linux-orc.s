# REQUIRES: system-linux

## Check that BOLT correctly updates ORC unwind information used by the Linux
## kernel.

# RUN: llvm-mc -filetype=obj -triple x86_64-unknown-unknown %s -o %t.o
# RUN: %clang %cflags -nostdlib %t.o -o %t.exe \
# RUN:   -Wl,--image-base=0xffffffff80000000,--no-dynamic-linker,--no-eh-frame-hdr

## Verify reading contents of ORC sections.

# RUN: llvm-bolt %t.exe --dump-orc -o /dev/null 2>&1 | FileCheck %s \
# RUN:   --check-prefix=CHECK-ORC

# CHECK-ORC: 	    BOLT-INFO: ORC unwind information:
# CHECK-ORC-NEXT: {sp: 8, bp: 0, info: 0x5}: _start
# CHECK-ORC-NEXT: {terminator}
# CHECK-ORC-NEXT: {sp: 8, bp: 0, info: 0x5}: foo
# CHECK-ORC-NEXT: {sp: 16, bp: -16, info: 0x15}: foo
# CHECK-ORC-NEXT: {sp: 16, bp: -16, info: 0x14}: foo
# CHECK-ORC-NEXT: {sp: 8, bp: 0, info: 0x5}: foo
# CHECK-ORC-NEXT: {terminator}
# CHECK-ORC-NEXT: {terminator}
# CHECK-ORC-NEXT: {terminator}


## Verify ORC bindings to instructions.

# RUN: llvm-bolt %t.exe --print-normalized --dump-orc --print-orc -o %t.out \
# RUN:   --keep-nops=0 --bolt-info=0 2>&1 | FileCheck %s


## Verify ORC bindings after rewrite.

# RUN: llvm-bolt %t.out -o %t.out.1 --print-normalized --print-orc \
# RUN:   2>&1 | FileCheck %s

## Verify ORC binding after rewrite when some of the functions are skipped.

# RUN: llvm-bolt %t.exe -o %t.out --skip-funcs=bar --bolt-info=0 --keep-nops=0
# RUN: llvm-bolt %t.out -o %t.out.1 --print-normalized --print-orc \
# RUN:   2>&1 | FileCheck %s

# CHECK:      BOLT-INFO: Linux kernel binary detected
# CHECK:      BOLT-INFO: parsed 9 ORC entries

  .text
  .globl _start
  .type _start, %function
_start:
# CHECK: Binary Function "_start"

  call foo
# CHECK:      callq foo           # ORC: {sp: 8, bp: 0, info: 0x5}
  ret
  .size _start, .-_start

  .globl foo
  .type foo, %function
foo:
# CHECK: Binary Function "foo"

  push %rbp
# CHECK:      pushq   %rbp        # ORC: {sp: 8, bp: 0, info: 0x5}
.L1:
  mov %rsp, %rbp
# CHECK:      movq    %rsp, %rbp  # ORC: {sp: 16, bp: -16, info: 0x15}
.L2:
  pop %rbp
# CHECK:      popq    %rbp        # ORC: {sp: 16, bp: -16, info: 0x14}
  nop
.L3:
  ret
# CHECK:      retq                # ORC: {sp: 8, bp: 0, info: 0x5}
  .size foo, .-foo

  .globl bar
  .type bar, %function
bar:
# CHECK:   Binary Function "bar"
	ret
## Same ORC info propagated from foo above.
# CHECK:      retq                # ORC: {sp: 8, bp: 0, info: 0x5}
.L4:
  .size bar, .-bar

# CHECK: BOLT-WARNING: Linux kernel support is experimental

  .section .orc_unwind,"a",@progbits
  .align 4
  .section .orc_unwind_ip,"a",@progbits
  .align 4

## ORC for _start.
  .section .orc_unwind
  .2byte 8
  .2byte 0
  .2byte 5
  .section .orc_unwind_ip
  .long _start - .

  .section .orc_unwind
  .2byte 0
  .2byte 0
  .2byte 0
  .section .orc_unwind_ip
  .long foo - .

## ORC for foo.
  .section .orc_unwind
  .2byte 8
  .2byte 0
  .2byte 5
  .section .orc_unwind_ip
  .long foo - .

  .section .orc_unwind
  .2byte 16
  .2byte -16
  .2byte 21
  .section .orc_unwind_ip
  .long .L1 - .

  .section .orc_unwind
  .2byte 16
  .2byte -16
  .2byte 20
  .section .orc_unwind_ip
  .long .L2 - .

  .section .orc_unwind
  .2byte 8
  .2byte 0
  .2byte 5
  .section .orc_unwind_ip
  .long .L3 - .

  .section .orc_unwind
  .2byte 0
  .2byte 0
  .2byte 0
  .section .orc_unwind_ip
  .long .L4 - .

## Duplicate terminator entries to test ORC reader.
  .section .orc_unwind
  .2byte 0
  .2byte 0
  .2byte 0
  .section .orc_unwind_ip
  .long .L4 - .

  .section .orc_unwind
  .2byte 0
  .2byte 0
  .2byte 0
  .section .orc_unwind_ip
  .long .L4 - .

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
