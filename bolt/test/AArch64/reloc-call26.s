## This test checks processing of R_AARCH64_CALL26 relocation
## when option `--funcs` is enabled

## We want to test on relocations against functions with both higher
## and lower addresses. The '--force-patch' option is used to prevent
## the functions func1 and func2 from being optimized, so that their
## addresses can remain unchanged. Therefore, the relocations can be
## updated via encodeValueAArch64 and the address order in the output
## binary is func1 < _start < func2.

# REQUIRES: system-linux

# RUN: llvm-mc -filetype=obj -triple aarch64-unknown-unknown \
# RUN:   %s -o %t.o
# RUN: %clang %cflags %t.o -o %t.exe -Wl,-q
# RUN: llvm-bolt %t.exe -o %t.bolt --funcs=func1,func2 \
# RUN:   --force-patch 2>&1 | FileCheck %s -check-prefix=CHECK-BOLT
# RUN: llvm-objdump -d --disassemble-symbols='_start' %t.bolt | \
# RUN:   FileCheck %s
# RUN: llvm-nm --numeric-sort --extern-only %t.bolt  | FileCheck \
# RUN:   %s -check-prefix=CHECK-FUNC-ORDER

# CHECK-BOLT: BOLT-WARNING: failed to patch entries in func1. The function will not be optimized.
# CHECK-BOLT: BOLT-WARNING: failed to patch entries in func2. The function will not be optimized.
# CHECK: {{.*}} bl {{.*}} <func1>
# CHECK: {{.*}} bl {{.*}} <func2>

# CHECK-FUNC-ORDER: {{.*}} func1
# CHECK-FUNC-ORDER-NEXT: {{.*}} _start
# CHECK-FUNC-ORDER-NEXT: {{.*}} func2

  .text
  .align 4
  .global func1
  .type func1, %function
func1:
  ret
  .size func1, .-func1
  .global _start
  .type _start, %function
_start:
  bl func1
  bl func2
  mov     w8, #93
  svc     #0
  .size _start, .-_start
  .global func2
  .type func2, %function
func2:
  ret
  .size func2, .-func2
