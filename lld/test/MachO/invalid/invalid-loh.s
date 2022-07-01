# REQUIRES: aarch64

# RUN: rm -rf %t; split-file %s %t
# RUN: llvm-mc -filetype=obj -triple=arm64-apple-darwin %t/section.s -o %t/section.o
# RUN: llvm-mc -filetype=obj -triple=arm64-apple-darwin %t/far.s -o %t/far.o
# RUN: not %lld -arch arm64 %t/section.o -o /dev/null 2>&1 | FileCheck %s --check-prefix=SECTION
# RUN: not %lld -arch arm64 %t/far.o -o /dev/null 2>&1 | FileCheck %s --check-prefix=FAR

# SECTION: error: Linker optimization hint spans multiple sections
# FAR:     error: Linker optimization hint at offset 0 has addresses too far apart

#--- section.s
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

#--- far.s
.globl _main
_main:
L1:
  adrp x0, _target@PAGE
  .zero 0x8000
L2:
  add  x0, x0, _target@PAGEOFF

_target:

.loh AdrpAdd L1, L2
.subsections_via_symbols
