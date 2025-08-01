## Check that llvm-bolt successfully relaxes branches for compact (<128MB) code
## model.

# RUN: llvm-mc -filetype=obj -triple aarch64-unknown-unknown %s -o %t.o
# RUN: link_fdata %s %t.o %t.fdata
# RUN: llvm-strip --strip-unneeded %t.o
# RUN: %clang %cflags %t.o -o %t.exe -Wl,-q -static
# RUN: llvm-bolt %t.exe -o %t.bolt --data %t.fdata --split-functions \
# RUN:   --keep-nops --compact-code-model
# RUN: llvm-objdump -d \
# RUN:   --disassemble-symbols=_start,_start.cold.0,foo,foo.cold.0 %t.bolt \
# RUN:   | FileCheck %s
# RUN: llvm-nm -nS %t.bolt | FileCheck %s --check-prefix=CHECK-NM

## Fragments of _start and foo will be separated by large_function which is over
## 1MB in size - larger than all conditional branches can cover requiring branch
## relaxation.

# CHECK-NM: _start
# CHECK-NM: foo
# CHECK-NM: 0000000000124f84 T large_function
# CHECK-NM: _start.cold.0
# CHECK-NM: foo.cold.0

  .text
  .globl _start
  .type _start, %function
_start:
# CHECK: <_start>:
# FDATA: 0 [unknown] 0 1 _start 0 0 100
  .cfi_startproc
  cmp  x0, 1
  b.eq  .L0
# CHECK: b.eq
# CHECK-NEXT: b
# CHECK-NEXT: b

  bl large_function
.L0:
  ret  x30
  .cfi_endproc
.size _start, .-_start

## Check that long branch in foo() is reused during relaxation. I.e. we should
## see just one branch to the cold fragment.

  .globl foo
  .type foo, %function
foo:
# CHECK: <foo>:
# FDATA: 0 [unknown] 0 1 foo 0 0 100
  .cfi_startproc
  cmp x0, 0
.T0:
  b.eq .ERROR
# CHECK: b {{.*}} <foo.cold.0>
# CHECK-NOT: b {{.*}} <foo.cold.0>
# FDATA: 1 foo #.T0# 1 foo #.T1# 0 100
.T1:
  bl large_function
  cmp x0, 1
.T2:
  b.eq .ERROR
# FDATA: 1 foo #.T2# 1 foo #.T3# 0 100
.T3:
  mov x1, x0
  mov x0, 0
  ret x30

# CHECK: <foo.cold.0>:
# CHECK-NEXT: mov x0, #0x1
# CHECK-NEXT: ret
.ERROR:
  mov x0, 1
  ret x30
  .cfi_endproc
.size foo, .-foo

  .globl large_function
  .type large_function, %function
large_function:
# FDATA: 0 [unknown] 0 1 large_function 0 0 100
  .cfi_startproc
  .rept 300000
    nop
  .endr
  ret  x30
  .cfi_endproc
.size large_function, .-large_function

## Force relocation mode.
  .reloc 0, R_AARCH64_NONE
