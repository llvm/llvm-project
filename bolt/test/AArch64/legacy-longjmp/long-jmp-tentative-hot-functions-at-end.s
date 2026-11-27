## Check that the LongJmp tentative layout matches the emitted layout when
## functions are split, with and without --hot-functions-at-end. With a profile
## and --lite=0, U goes to .text.cold together with the cold fragment of H.
## With --hot-functions-at-end, .text.cold comes first and starts at the layout
## start address, aligned to its own alignment rather than that of .text.

# REQUIRES: asserts

# RUN: llvm-mc -filetype=obj -triple=aarch64-linux %s -o %t.o
# RUN: ld.lld %t.o -o %t.exe -q --entry=H
# RUN: link_fdata --no-lbr %s %t.exe %t.fdata
# RUN: llvm-strip --strip-unneeded %t.exe

# RUN: llvm-bolt %t.exe -o %t.bolt --data %t.fdata --split-functions --lite=0 \
# RUN:   -debug-only=longjmp 2> %t.log
# RUN: echo "EMITTED:" >> %t.log
# RUN: llvm-nm %t.bolt >> %t.log
# RUN: FileCheck %s --input-file=%t.log

# RUN: llvm-bolt %t.exe -o %t.hfe.bolt --data %t.fdata --split-functions \
# RUN:   --lite=0 --hot-functions-at-end -debug-only=longjmp 2> %t.hfe.log
# RUN: echo "EMITTED:" >> %t.hfe.log
# RUN: llvm-nm %t.hfe.bolt >> %t.hfe.log
# RUN: FileCheck %s --input-file=%t.hfe.log

# CHECK-DAG: {{^}}U tentative: [[#%x,U:]]
# CHECK-DAG: {{^}}H tentative: [[#%x,H:]]
# CHECK-DAG: {{^}}H cold tentative: [[#%x,HCOLD:]]
# CHECK:     EMITTED:
# CHECK-DAG: [[#%.16x,U]] T U{{$}}
# CHECK-DAG: [[#%.16x,H]] T H{{$}}
# CHECK-DAG: [[#%.16x,HCOLD]] t H.cold.0{{$}}

  .text
  .globl U
  .type U, %function
U:
  .rept 5
  add x0, x0, #1
  .endr
  ret
  .size U, .-U

  .globl H
  .type H, %function
H:
.H_entry:
# FDATA: 1 H #.H_entry# 100
  cbz x0, .H_ret
  b .H_cold
.H_ret:
# FDATA: 1 H #.H_ret# 100
  ret
.H_cold:
  mov x0, #1
  b .H_ret
  .size H, .-H

## Force relocation mode.
  .reloc 0, R_AARCH64_NONE
