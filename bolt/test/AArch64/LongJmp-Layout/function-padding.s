## Check that LongJmp includes padding before and after functions in its
## layout. Either padding request puts target outside the range of
## foo's direct call and requires a stub.

# REQUIRES: system-linux

# RUN: llvm-mc -filetype=obj -triple aarch64-unknown-unknown %s -o %t.o
# RUN: %clang %cflags %t.o -o %t.exe -nostdlib -Wl,-q
# RUN: link_fdata --no-lbr %s %t.exe %t.fdata
# RUN: llvm-strip --strip-unneeded %t.exe
# RUN: llvm-bolt %t.exe -o %t.after.bolt --data %t.fdata --lite=0 \
# RUN:   --pad-funcs=foo:134217728 2>&1 | FileCheck %s --check-prefix=MAIN
# RUN: llvm-bolt %t.exe -o %t.before.bolt --data %t.fdata --lite=0 \
# RUN:   --pad-funcs-before=target:134217728 2>&1 \
# RUN:   | FileCheck %s --check-prefix=MAIN
# RUN: llvm-bolt %t.exe -o %t.split-before.bolt --data %t.fdata --lite=0 \
# RUN:   --split-functions --split-all-cold \
# RUN:   --pad-funcs-before=target:134217728 2>&1 \
# RUN:   | FileCheck %s --check-prefix=SPLIT-BEFORE
# RUN: llvm-bolt %t.exe -o %t.split-after.bolt --data %t.fdata --lite=0 \
# RUN:   --split-functions --split-all-cold --pad-funcs=foo:134217728 2>&1 \
# RUN:   | FileCheck %s --check-prefix=SPLIT-AFTER

# MAIN: BOLT-INFO: Inserted 1 stubs in the hot area and 0 stubs in the cold area.
# SPLIT-BEFORE: BOLT-INFO: Inserted 0 stubs in the hot area and 1 stubs in the cold area.
# SPLIT-AFTER: BOLT-INFO: Inserted 1 stubs in the hot area and 1 stubs in the cold area.

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
  ret
  .size target, .-target

  .globl _start
  .type _start, %function
_start:
  nop
  ret

  .size _start, .-_start
