## Check that llvm-bolt handles code larger than 256MB.

# RUN: llvm-mc -filetype=obj -triple aarch64-unknown-unknown %s -o %t.o
# RUN: link_fdata %s %t.o %t.fdata
# RUN: llvm-strip --strip-unneeded %t.o
# RUN: %clang %cflags %t.o -o %t.exe -nostdlib -Wl,-q
# RUN: llvm-bolt %t.exe -o %t.bolt --relax-exp --lite=0
# RUN: llvm-bolt %t.exe -o %t.bolt --relax-exp --lite=1 --data %t.fdata \
# RUN:   --print-normalized 2>&1 | FileCheck %s --check-prefix=CHECK-VENEER
# RUN: llvm-bolt %t.exe -o %t.bolt --relax-exp --hot-functions-at-end --lite=0 \
# RUN:   --data %t.fdata
# RUN: llvm-objdump -d %t.bolt | FileCheck %s

## The constant island at the end of a function makes it ~112MB in size.
## Thus the total code size exceeds 300MB.

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

## Check that BOLT-introduced veneers have proper names.
# CHECK-LABEL: <hot>:
# CHECK-NEXT: bl {{.*}} <__AArch64ADRPThunk_foo>
# CHECK-NEXT: bl {{.*}} <__AArch64Thunk_bar>
# CHECK-NEXT: bl {{.*}} <_start>

  .global _start
  .type _start, %function
_start:
  bl foo
  bl bar
  bl hot
  ret
  .space 0x7000000
  .size _start, .-_start

