## Check that llvm-bolt correctly handles veneers in lite mode.

# RUN: llvm-mc -filetype=obj -triple aarch64-unknown-unknown %s -o %t.o
# RUN: link_fdata %s %t.o %t.fdata
# RUN: llvm-strip --strip-unneeded %t.o
# RUN: %clang %cflags %t.o -o %t.exe -nostdlib -Wl,-q
# RUN: llvm-bolt %t.exe -o %t.bolt --lite=1 --data %t.fdata \
# RUN:   --print-normalized 2>&1 | FileCheck %s --check-prefix=CHECK-VENEER

## Constant islands at the end of functions foo(), bar(), and _start() make each
## one of them ~112MB in size. Thus the total code size exceeds 300MB.

  .text
  .global foo
  .type foo, %function
foo:
  bl _start
  bl bar
  ret
  .space 0x7000000
  .size foo, .-foo

  .global bar
  .type bar, %function
bar:
  bl foo
  bl _start
  ret
  .space 0x7000000
  .size bar, .-bar

  .global hot
  .type hot, %function
hot:
# FDATA: 0 [unknown] 0 1 hot 0 0 100
  bl foo
  bl bar
  bl _start
  ret
  .size hot, .-hot

## Check that BOLT sees the call to foo, not to its veneer.
# CHECK-VENEER-LABEL: Binary Function "hot"
# CHECK-VENEER: bl
# CHECK-VENEER-SAME: {{[[:space:]]foo[[:space:]]}}

  .global _start
  .type _start, %function
_start:
  bl foo
  bl bar
  bl hot
  ret
  .space 0x7000000
  .size _start, .-_start

