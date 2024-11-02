# This reproduces a bug where BOLT would read __hot_end as
# the name of a function even when -hot-text is used, which
# means BOLT will emit another __hot_end label, eventually
# asserting due to a symbol redefinition in MCStreamer.

# REQUIRES: system-linux

# RUN: llvm-mc -filetype=obj -triple x86_64-unknown-unknown \
# RUN:   %s -o %t.o
# RUN: link_fdata %s %t.o %t.fdata
# RUN: llvm-strip --strip-unneeded %t.o
# RUN: %clang %cflags -no-pie %t.o -o %t.exe -Wl,-q

# RUN: llvm-bolt %t.exe --relocs=1 --hot-text --reorder-functions=hfsort \
# RUN:    --data %t.fdata -o %t.out | FileCheck %s

# RUN: %t.out 1

# CHECK: BOLT-INFO: setting __hot_end to

# RUN: llvm-nm -n %t.exe | FileCheck %s --check-prefix=CHECK-INPUT
# RUN: llvm-nm -n %t.out | FileCheck %s --check-prefix=CHECK-OUTPUT

# CHECK-INPUT:       __hot_start
# CHECK-INPUT-NEXT:  main
# CHECK-INPUT-NEXT:  foo
# CHECK-INPUT-NEXT:  __hot_end

# Our fdata only logs activity in main, so hot markers will change
# CHECK-OUTPUT:       __hot_start
# CHECK-OUTPUT-NEXT:  main
# CHECK-OUTPUT-NEXT:  __hot_end

  .text
  .globl  main
  .type main, %function
  .globl  __hot_start
  .type __hot_start, %object
  .p2align  4
main:
__hot_start:
# FDATA: 0 [unknown] 0 1 main 0 0 510
  pushq %rbp
  movq  %rsp, %rbp
  cmpl  $0x2, %edi
  jb    .BBend
.BB2:
  callq bar
  leaq mystring, %rdi
  callq puts

.BBend:
  xorq %rax, %rax
  leaveq
  retq
  .size main, .-main

  .globl foo
  .type foo, %function
  .p2align 4
foo:
  retq
  .size foo, .-foo

  .globl __hot_end
  .type __hot_end, %object
  .p2align 2
__hot_end:
  int3
  .size __hot_end, 0

  .globl bar
  .type bar, %function
  .p2align 4
bar:
  retq
  .size bar, .-bar

  .data
mystring: .asciz "test\n"
