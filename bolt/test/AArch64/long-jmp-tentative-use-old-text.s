## Check that the LongJmp tentative layout matches the emitted layout with
## --use-old-text. The new code is placed at the start of the original .text,
## or at its end with --hot-functions-at-end, if it fits there once its start is
## aligned. With --align-text=0x4000 it no longer fits, and is placed in the new
## code segment instead.

# REQUIRES: asserts

# RUN: llvm-mc -filetype=obj -triple=aarch64-linux %s -o %t.o
# RUN: ld.lld %t.o -o %t.exe -q --entry=H
# RUN: link_fdata --no-lbr %s %t.exe %t.fdata
# RUN: llvm-strip --strip-unneeded %t.exe

# RUN: llvm-bolt %t.exe -o %t.bolt --data %t.fdata --use-old-text \
# RUN:   --align-text=4 -debug-only=longjmp > %t.log 2>&1
# RUN: echo "EMITTED:" >> %t.log
# RUN: llvm-nm %t.bolt >> %t.log
# RUN: FileCheck %s --input-file=%t.log
# RUN: FileCheck %s --input-file=%t.log --check-prefix=FITS

# RUN: llvm-bolt %t.exe -o %t.hfe.bolt --data %t.fdata --use-old-text \
# RUN:   --align-text=4 --hot-functions-at-end -debug-only=longjmp \
# RUN:   > %t.hfe.log 2>&1
# RUN: echo "EMITTED:" >> %t.hfe.log
# RUN: llvm-nm %t.hfe.bolt >> %t.hfe.log
# RUN: FileCheck %s --input-file=%t.hfe.log
# RUN: FileCheck %s --input-file=%t.hfe.log --check-prefix=FITS

# RUN: llvm-bolt %t.exe -o %t.new.bolt --data %t.fdata --use-old-text \
# RUN:   --align-text=0x4000 -debug-only=longjmp > %t.new.log 2>&1
# RUN: echo "EMITTED:" >> %t.new.log
# RUN: llvm-nm %t.new.bolt >> %t.new.log
# RUN: FileCheck %s --input-file=%t.new.log
# RUN: FileCheck %s --input-file=%t.new.log --check-prefix=NOFIT

# FITS: BOLT-INFO: using original .text for new code
# NOFIT: BOLT-WARNING: --use-old-text failed

# CHECK-DAG: {{^}}H tentative: [[#%x,H:]]
# CHECK-DAG: {{^}}H2 tentative: [[#%x,H2:]]
# CHECK-DAG: {{^}}U tentative: [[#%x,U:]]
# CHECK:     EMITTED:
# CHECK-DAG: [[#%.16x,H]] T H{{$}}
# CHECK-DAG: [[#%.16x,H2]] T H2{{$}}
# CHECK-DAG: [[#%.16x,U]] T U{{$}}

## The input functions are 4KB aligned, so the original .text has room for the
## new code.
  .text
  .globl H
  .type H, %function
H:
.H_entry:
# FDATA: 1 H #.H_entry# 100
  .rept 20
  add x0, x0, #1
  .endr
  ret
  .size H, .-H

  .p2align 12
  .globl H2
  .type H2, %function
H2:
.H2_entry:
# FDATA: 1 H2 #.H2_entry# 100
  ret
  .size H2, .-H2

  .p2align 12
  .globl U
  .type U, %function
U:
  ret
  .space 0x1000
  .size U, .-U

## Force relocation mode.
  .reloc 0, R_AARCH64_NONE
