## Same failure as pcrel-patch-skipped-func.s, but reached through the default
## AArch64 configuration: lite mode. Functions without profile data are ignored
## (RewriteInstance.cpp, shouldProcess()), so a non-profiled function holding an
## assembler-resolved ADR to a hot function produces an instruction patch that
## cannot be encoded once the hot function moves.
##
## No --skip-funcs here: only a profile.
##
## https://github.com/llvm/llvm-project/issues/194938

# REQUIRES: system-linux

# RUN: llvm-mc -filetype=obj -triple aarch64-unknown-unknown %s -o %t.o
# RUN: %clang %cflags %t.o -o %t.exe -nostdlib -fuse-ld=lld -Wl,-q
# RUN: echo "0 [unknown] 0 1 target_fn/1 0 0 100" > %t.fdata
# RUN: llvm-bolt %t.exe -o %t.bolt --data %t.fdata --lite 2>&1 \
# RUN:   | FileCheck %s --check-prefix=CHECK-BOLT
# RUN: llvm-objdump -d --disassemble-symbols=victim %t.bolt | FileCheck %s

# CHECK-BOLT: BOLT-WARNING: unable to update PC-relative reference to
# CHECK-BOLT-NOT: BOLT-ERROR

## victim is not emitted; its ADR is preserved.
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

## Local symbol, so the ADR below is resolved by the assembler and no
## R_AARCH64_ADR_PREL_LO21 relocation is emitted for it.
  .type target_fn, %function
target_fn:
  mul w0, w0, w0
  add w0, w0, #1
  ret
  .size target_fn, .-target_fn

## No profile data: ignored in lite mode, but its function pointer
## materialization still references a function BOLT moves.
  .globl victim
  .type victim, %function
victim:
  adr x8, target_fn
  blr x8
  ret
  .size victim, .-victim
