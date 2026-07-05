## This test verifies that no unnecessary stubs are inserted when each DotAddress increases during a lookup.

# REQUIRES: system-linux, asserts

# RUN: llvm-mc -filetype=obj -triple aarch64-unknown-unknown %s -o %t.o
# RUN: %clang %cflags -O0 %t.o -o %t.exe -Wl,-q
# RUN: link_fdata %s %t.o %t.fdata
# RUN: llvm-bolt %t.exe -o %t.bolt  \
# RUN:   --data %t.fdata  | FileCheck %s

# CHECK: BOLT-INFO: Inserted 1 stubs in the hot area and 0 stubs in the cold area. 

  .section .text
  .global _start
  .global far_away_func

  .align 4
  .global _start
  .type _start, %function
_start:
# FDATA: 0 [unknown] 0 1 _start 0 0 100
    bl far_away_func
    bl far_away_func
    ret  
  .space 0x8000000
  .global far_away_func
  .type far_away_func, %function
far_away_func:
    add x0, x0, #1
    ret

.reloc 0, R_AARCH64_NONE
