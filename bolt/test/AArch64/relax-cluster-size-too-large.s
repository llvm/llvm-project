## Experimental relaxation rejects cluster sizes above 126 MiB.

# REQUIRES: system-linux

# RUN: %clang %cflags -Wl,-q -Wl,-e,A %s -o %t -nostdlib
# RUN: not llvm-bolt %t -o %t.bolt --relax-exp \
# RUN:   --max-cluster-size=132120580 > %t.log 2>&1
# RUN: FileCheck %s < %t.log

# CHECK: FATAL BOLT-ERROR: --max-cluster-size must be at most 126 MiB

  .text
  .globl A
  .type A, %function
A:
  ret
  .size A, .-A

## Force relocation mode.
  .reloc 0, R_AARCH64_NONE
