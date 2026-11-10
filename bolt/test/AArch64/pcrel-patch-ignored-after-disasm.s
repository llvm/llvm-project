## Same failure as pcrel-patch-skipped-func.s, but the function is ignored by
## BOLT's own decision *after* it was disassembled, so the scan happens in
## BinaryFunction::setIgnored() rather than in
## RewriteInstance::disassembleFunctions(). A guard placed at the latter call
## site does not cover this path.
##
## Here a data pointer into the middle of one of victim's instructions makes
## BOLT report corrupted control flow and ignore the function. No BOLT options
## are needed at all.
##
## https://github.com/llvm/llvm-project/issues/194938

# REQUIRES: system-linux

# RUN: llvm-mc -filetype=obj -triple aarch64-unknown-unknown %s -o %t.o
# RUN: %clang %cflags %t.o -o %t.exe -nostdlib -fuse-ld=lld -Wl,-q
# RUN: llvm-bolt %t.exe -o %t.bolt 2>&1 | FileCheck %s --check-prefix=CHECK-BOLT
# RUN: llvm-objdump -d --disassemble-symbols=victim %t.bolt | FileCheck %s

## The function is dropped from processing, but BOLT must not fail.
# CHECK-BOLT: corrupted control flow detected in function victim
# CHECK-BOLT: BOLT-WARNING: unable to update PC-relative reference to
# CHECK-BOLT-NOT: BOLT-ERROR

# CHECK-LABEL: <victim>:
# CHECK: adr x8

  .text
  .globl _start
  .type _start, %function
_start:
  mov w0, #5
  bl target_fn
  bl victim
  mov w8, #93
  svc #0
  .size _start, .-_start

## Local symbol: the ADR below carries no relocation.
  .type target_fn, %function
target_fn:
  mul w0, w0, w0
  add w0, w0, #1
  ret
  .size target_fn, .-target_fn

  .globl victim
  .type victim, %function
victim:
  adr x8, target_fn
  blr x8
  ret
  .size victim, .-victim

## Pointer into the middle of victim's first instruction.
  .data
  .p2align 3
badptr:
  .quad victim+2
