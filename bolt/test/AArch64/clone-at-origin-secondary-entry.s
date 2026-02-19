## Test that --clone-at-origin creates clone symbols for secondary entry points.

# REQUIRES: system-linux

# RUN: %clang %cflags -nostdlib %s -o %t.exe -Wl,-q
# RUN: llvm-bolt %t.exe -o %t.bolt --clone-at-origin

## Get the original alt_entry address.
# RUN: llvm-nm %t.exe | grep " alt_entry$" | cut -d' ' -f1 > %t.alt_orig

## Verify clone symbols exist for both main function and secondary entry point.
# RUN: llvm-nm -n %t.bolt | FileCheck %s

# CHECK: main_func.clone.0
# CHECK: alt_entry.clone.0

## Verify alt_entry clone address matches original.
# RUN: llvm-nm %t.bolt | grep "alt_entry.clone.0" | cut -d' ' -f1 > %t.alt_clone
# RUN: diff %t.alt_orig %t.alt_clone

  .text
  .globl main_func
  .type main_func, %function
main_func:
  nop
  .type alt_entry, %function
alt_entry:
  ret
  .size main_func, .-main_func

  .globl _start
  .type _start, %function
_start:
  bl main_func
  ret
  .size _start, .-_start
