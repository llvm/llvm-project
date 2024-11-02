# REQUIRES: system-linux

## Check that BOLT correctly reads ORC unwind information used by Linux kernel.

# RUN: llvm-mc -filetype=obj -triple x86_64-unknown-unknown %s -o %t.o
# RUN: %clang %cflags -nostdlib %t.o -o %t.exe -Wl,--image-base=0xffffffff80000000,--no-dynamic-linker,--no-eh-frame-hdr

# RUN: llvm-bolt %t.exe --print-normalized --dump-orc --print-orc -o %t.out \
# RUN:   |& FileCheck %s

# CHECK:      BOLT-INFO: Linux kernel binary detected
# CHECK: 			BOLT-INFO: ORC unwind information:
# CHECK-NEXT: {sp: 8, bp: 0, info: 0x5}: _start
# CHECK-NEXT: {sp: 0, bp: 0, info: 0x0}: _start
# CHECK-NEXT: {sp: 8, bp: 0, info: 0x5}: foo
# CHECK-NEXT: {sp: 16, bp: -16, info: 0x15}: foo
# CHECK-NEXT: {sp: 16, bp: -16, info: 0x14}: foo
# CHECK-NEXT: {sp: 8, bp: 0, info: 0x5}: foo
# CHECK-NEXT: {sp: 0, bp: 0, info: 0x0}: bar
# CHECK-NEXT: {sp: 0, bp: 0, info: 0x0}: bar

  .text
  .globl _start
  .type _start, %function
_start:

  call foo
# CHECK:      callq foo           # ORC: {sp: 8, bp: 0, info: 0x5}
  ret
  .size _start, .-_start

  .globl foo
  .type foo, %function
foo:
  push %rbp
# CHECK:      pushq   %rbp        # ORC: {sp: 8, bp: 0, info: 0x5}
.L1:
  mov %rsp, %rbp
# CHECK:      movq    %rsp, %rbp  # ORC: {sp: 16, bp: -16, info: 0x15}
.L2:
  pop %rbp
# CHECK:      popq    %rbp        # ORC: {sp: 16, bp: -16, info: 0x14}
.L3:
  ret
# CHECK:      retq                # ORC: {sp: 8, bp: 0, info: 0x5}
  .size foo, .-foo

bar:
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

## Fake Linux Kernel sections.
  .section __ksymtab,"a",@progbits
  .section __ksymtab_gpl,"a",@progbits
