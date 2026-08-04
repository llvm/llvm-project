## An execution-cold basic block remains in the main fragment when function
## splitting is disabled. Check that a long-jump stub originating in that block
## is classified and laid out in the main (hot) area.

# REQUIRES: system-linux, asserts, target=aarch64{{.*}}

# RUN: %clang %cflags -Wl,-q %s -o %t
# RUN: link_fdata --no-lbr %s %t %t.fdata
# RUN: llvm-strip --strip-unneeded %t
# RUN: llvm-bolt %t -o %t.unsplit.bolt --data %t.fdata --lite=0 \
# RUN:   --align-text=0x10000000 --skip-funcs=target 2>&1 \
# RUN:   | FileCheck %s --check-prefix=UNSPLIT
# RUN: llvm-bolt %t -o %t.split.bolt --data %t.fdata --lite=0 \
# RUN:   --split-functions --split-all-cold --align-text=0x10000000 \
# RUN:   --skip-funcs=target 2>&1 | FileCheck %s --check-prefix=SPLIT

# UNSPLIT: BOLT-INFO: Inserted 1 stubs in the hot area and 0 stubs in the cold area.
# SPLIT: BOLT-INFO: Inserted 0 stubs in the hot area and 1 stubs in the cold area.

  .text
  .globl foo
  .type foo, %function
foo:
.entry_foo:
# FDATA: 1 foo #.entry_foo# 10
  cbnz x0, .hot_foo
.Lcold:
  bl target
  ret
.hot_foo:
# FDATA: 1 foo #.hot_foo# 10
  ret
  .size foo, .-foo

  .globl target
  .type target, %function
target:
  nop
  ret
  .size target, .-target

  .globl main
  .type main, %function
main:
  mov x0, xzr
  bl foo

  ret
  .size main, .-main
