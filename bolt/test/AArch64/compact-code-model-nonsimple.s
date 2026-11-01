## Check that llvm-bolt relaxes conditional tail calls in non-simple functions
## for compact code model. Without the relaxation, the branches below are out
## of range after reordering and llvm-bolt fails with JITLink error.

# REQUIRES: system-linux

# RUN: llvm-mc -filetype=obj -triple aarch64-unknown-unknown %s -o %t.o
# RUN: llvm-strip --strip-unneeded %t.o
# RUN: %clang %cflags %t.o -o %t.exe -Wl,-q -static
# RUN: echo nonsimple > %t.order
# RUN: echo large_function >> %t.order
# RUN: echo _start >> %t.order
# RUN: llvm-bolt %t.exe -o %t.bolt --compact-code-model --keep-nops \
# RUN:   --function-order=%t.order --print-cfg --print-only=nonsimple \
# RUN:   | FileCheck %s --check-prefix=CHECK-CFG
# RUN: llvm-objdump -d --disassemble-symbols=nonsimple %t.bolt | FileCheck %s

  .text
  .globl _start
  .type _start, %function
_start:
  .cfi_startproc
  bl nonsimple
  ret x30
  .cfi_endproc
.size _start, .-_start

## 64KB of code placed between "nonsimple" and "cold_target" by the order file,
## which puts "cold_target" beyond the +-32KB reach of the tbz below.
  .globl large_function
  .type large_function, %function
large_function:
  .cfi_startproc
  .rept 16000
    nop
  .endr
  ret x30
  .cfi_endproc
.size large_function, .-large_function

## Non-simple function ("br x16" has unknown control flow) with two conditional
## tail calls to the same target.
  .globl nonsimple
  .type nonsimple, %function
nonsimple:
  .cfi_startproc
  cmp x0, #1
  b.eq cold_target
  tbz x0, #0, cold_target
  ldr x16, [sp]
  br x16
  .cfi_endproc
.size nonsimple, .-nonsimple

  .globl cold_target
  .type cold_target, %function
cold_target:
  .cfi_startproc
  mov x0, #1
  ret x30
  .cfi_endproc
.size cold_target, .-cold_target

## Force relocation mode.
  .reloc 0, R_AARCH64_NONE

## Verify that the function under test is really non-simple. If this ever
## starts printing "IsSimple : 1", the test no longer covers the non-simple
## relaxation path.
# CHECK-CFG: Binary Function "nonsimple" after building cfg
# CHECK-CFG: IsSimple    : 0

## Both branches should be retargeted to a single trampoline appended after the
## function body, which in turn branches to the original target. The layout of
## the function itself must be preserved.
##
## Note that the tbz reuses the trampoline created for the b.eq, as both target
## the same symbol: the address captured from the b.eq must match the one used
## by the tbz, and no second trampoline may be emitted.
# CHECK: <nonsimple>:
# CHECK:      cmp x0, #0x1
# CHECK-NEXT: b.eq 0x[[TRAMP:[0-9a-f]+]] <nonsimple+0x{{[0-9a-f]+}}>
# CHECK-NEXT: tbz {{.*}}, 0x[[TRAMP]] <nonsimple+0x{{[0-9a-f]+}}>
# CHECK-NEXT: ldr x16, [sp]
# CHECK-NEXT: br x16
# CHECK-NEXT: [[TRAMP]]: {{.*}} b 0x{{[0-9a-f]+}} <cold_target>
# CHECK-NOT: b 0x{{[0-9a-f]+}} <cold_target>
