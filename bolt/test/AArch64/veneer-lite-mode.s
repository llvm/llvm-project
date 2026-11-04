## Check that llvm-bolt correctly handles veneers in lite mode.
##
## Constant island at the end of function foo makes it larger than 128MB thus
## a veneer is needed to call foo from _start.
## Check that BOLT recognizes the veneer in lite mode even when the veneer is
## not covered by the profile.


# RUN: llvm-mc -filetype=obj -triple aarch64-unknown-unknown %s -o %t.o
# RUN: link_fdata %s %t.o %t.fdata
# RUN: llvm-strip --strip-unneeded %t.o
# RUN: %clang %cflags %t.o -o %t.exe -nostdlib -Wl,-q
# RUN: llvm-objdump --disassemble-symbols=_start %t.exe \
# RUN:   | FileCheck %s --check-prefix=CHECK-INPUT
# RUN: llvm-bolt %t.exe -o %t.bolt --lite=1 --data %t.fdata --print-normalized \
# RUN:   | FileCheck %s --check-prefix=CHECK-BOLT

  .text
  .global foo
  .type foo, %function
foo:
  ret
  .space 0x8000000
  .size foo, .-foo

  .global _start
  .type _start, %function
_start:
# FDATA: 0 [unknown] 0 1 _start 0 0 100
  bl foo
  ret
  .size _start, .-_start

# CHECK-INPUT-LABEL: <_start>:
# CHECK-INPUT-NEXT:    bl {{.*}} <__AArch64ADRPThunk_foo>

## Check that BOLT sees the call to foo, not to its veneer.
# CHECK-BOLT-LABEL: Binary Function "_start"
# CHECK-BOLT: bl
# CHECK-BOLT-SAME: {{[[:space:]]foo[[:space:]]}}
