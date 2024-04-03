# RUN: llvm-mc -filetype=obj -triple x86_64-unknown-unknown %s -o %t.o
# RUN: link_fdata %s %t.o %t.fdata
# RUN: llvm-strip --strip-unneeded %t.o
# RUN: %clang %cflags %t.o -o %t.exe
# RUN: llvm-bolt %t.exe --data %t.fdata -o %t.null | FileCheck %s

## Check that profile is correctly matched by functions with variable suffixes.
## E.g., LTO-generated name foo.llvm.123 should match foo.llvm.*.

# CHECK: 6 out of {{.*}} functions in the binary {{.*}} have non-empty execution profile
# CHECK-NOT: profile for {{.*}} objects was ignored

	.globl _start
_start:

LL_start_0:
# FDATA: 1 _start #LL_start_0# 1 foo.llvm.321 0 0 1
  call foo.llvm.123

LL_start_1:
# FDATA: 1 _start #LL_start_1# 1 foo.constprop.321 0 0 1
  call foo.constprop.123

LL_start_2:
# FDATA: 1 _start #LL_start_2# 1 foo.lto_priv.321 0 0 1
  call foo.lto_priv.123

LL_start_3:
# FDATA: 1 _start #LL_start_3# 1 foo.__uniq.321 0 0 1
  call foo.__uniq.123

LL_start_4:
# FDATA: 1 _start #LL_start_4# 1 foo.__uniq.654.llvm.321 0 0 1
  call foo.__uniq.456.llvm.123

  call exit
  .size _start, .-_start

  .globl foo.llvm.123
  .type foo.llvm.123,@function
foo.llvm.123:
  ret
  .size foo.llvm.123, .-foo.llvm.123

  .globl foo.constprop.123
  .type foo.constprop.123,@function
foo.constprop.123:
  ret
  .size foo.constprop.123, .-foo.constprop.123

  .globl foo.lto_priv.123
  .type foo.lto_priv.123,@function
foo.lto_priv.123:
  ret
  .size foo.lto_priv.123, .-foo.lto_priv.123

  .globl foo.__uniq.123
  .type foo.__uniq.123,@function
foo.__uniq.123:
  ret
  .size foo.__uniq.123, .-foo.__uniq.123

  .globl foo.__uniq.456.llvm.123
  .type foo.__uniq.456.llvm.123,@function
foo.__uniq.456.llvm.123:
  ret
  .size foo.__uniq.456.llvm.123, .-foo.__uniq.456.llvm.123
