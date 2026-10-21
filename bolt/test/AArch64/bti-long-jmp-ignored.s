# This test checks the situation where LongJmp adds a stub targeting an ignored (skipped) function.
# As far_away_func does not have a nop at the entry, BOLT cannot safely patch it.

# REQUIRES: system-linux, asserts

# RUN: llvm-mc -filetype=obj -triple aarch64-unknown-unknown \
# RUN:   -mattr=+bti -aarch64-mark-bti-property %s -o %t.o
# RUN: %clang %cflags -O0 %t.o -o %t.exe -Wl,-q -Wl,-z,force-bti
# RUN: not llvm-bolt %t.exe -o %t.bolt \
# RUN:   --align-text=0x10000000 --skip-funcs=far_away_func 2>&1 | FileCheck %s

# CHECK:  BOLT-ERROR: Cannot add BTI to function without CFG far_away_func. Recompile the binary using -fpatchable-function-entry 1 to include a nop at the entry

  .section .text
  .align 4
  .global _start
  .type _start, %function
_start:
    bti c
    bl far_away_func
    ret

# This is skipped, so it stays in the .bolt.org.text.
# The .text produced by BOLT is aligned to 0x10000000,
# so _start will need a stub to jump here.
  .global far_away_func
  .type far_away_func, %function
far_away_func:
    add x0, x0, #1
    ret

.reloc 0, R_AARCH64_NONE

