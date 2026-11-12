# REQUIRES: aarch64
# RUN: rm -rf %t; mkdir -p %t

# RUN: llvm-mc -filetype=obj -triple=arm64-apple-darwin %s -o %t/test.o
# RUN: not %lld -arch arm64 %t/test.o -o /dev/null 2>&1 | FileCheck %s

# CHECK: error: {{.*}}test.o: linker optimization hint spans multiple sections

.globl _main
_main:
L1:
  adrp x0, _target@PAGE

_foo:
L2:
  add x0, x0, _target@PAGEOFF

_target:

.loh AdrpAdd L1, L2
.subsections_via_symbols
