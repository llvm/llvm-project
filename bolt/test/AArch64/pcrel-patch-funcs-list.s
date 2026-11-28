## Same failure as pcrel-patch-skipped-func.s, reached through an inclusion list
## instead of an exclusion list: with --funcs, everything not named is ignored
## without going through mustSkip().
##
## https://github.com/llvm/llvm-project/issues/194938

# REQUIRES: system-linux

# RUN: llvm-mc -filetype=obj -triple aarch64-unknown-unknown %s -o %t.o
# RUN: %clang %cflags %t.o -o %t.exe -nostdlib -fuse-ld=lld -Wl,-q
# RUN: llvm-bolt %t.exe -o %t.bolt --funcs='target_fn.*' 2>&1 \
# RUN:   | FileCheck %s --check-prefix=CHECK-BOLT
# RUN: llvm-objdump -d --disassemble-symbols=victim %t.bolt | FileCheck %s

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
