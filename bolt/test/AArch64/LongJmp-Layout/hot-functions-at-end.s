## Check that LongJmp mirrors code-section placement when cold sections are
## placed before hot sections with --hot-functions-at-end, including the
## backward section allocation used by --use-old-text.

# REQUIRES: system-linux, asserts

# RUN: llvm-mc -filetype=obj -triple aarch64-unknown-unknown %s -o %t.o
# RUN: %clang %cflags %t.o -o %t.exe -nostdlib -Wl,-q
# RUN: link_fdata --no-lbr %s %t.exe %t.fdata
# RUN: llvm-bolt %t.exe -o %t.hot.bolt --data %t.fdata --lite=0 \
# RUN:   --reorder-functions=exec-count --hot-functions-at-end \
# RUN:   --align-text=16 --align-functions=4 --debug-only=longjmp \
# RUN:   > %t.hot.log 2>&1
# RUN: llvm-nm -n --format=posix %t.hot.bolt >> %t.hot.log
# RUN: FileCheck %s --check-prefix=HOT < %t.hot.log
# RUN: llvm-bolt %t.exe -o %t.old.bolt --data %t.fdata --lite=0 \
# RUN:   --use-old-text --reorder-functions=exec-count \
# RUN:   --hot-functions-at-end --align-text=4 --align-functions=4 \
# RUN:   --debug-only=longjmp > %t.old.log 2>&1
# RUN: llvm-nm -n --format=posix %t.old.bolt >> %t.old.log
# RUN: FileCheck %s --check-prefix=OLD < %t.old.log
# RUN: llvm-bolt %t.exe -o %t.old-forward.bolt --data %t.fdata --lite=0 \
# RUN:   --use-old-text --hot-text=0 --reorder-functions=exec-count \
# RUN:   --align-text=4 --align-functions=4 --debug-only=longjmp \
# RUN:   > %t.old-forward.log 2>&1
# RUN: llvm-nm -n --format=posix %t.old-forward.bolt >> %t.old-forward.log
# RUN: FileCheck %s --check-prefix=OLD-FORWARD < %t.old-forward.log
# RUN: llvm-bolt %t.exe -o %t.old-forward-fail.bolt --data %t.fdata \
# RUN:   --lite=0 --use-old-text --hot-text=0 \
# RUN:   --reorder-functions=exec-count --align-text=4 --align-functions=4 \
# RUN:   --pad-funcs=_start:1024 --debug-only=longjmp 2>&1 | \
# RUN:   FileCheck %s --check-prefix=OLD-FORWARD-FAIL
# RUN: llvm-bolt %t.exe -o %t.old-fail.bolt --data %t.fdata --lite=0 \
# RUN:   --use-old-text --reorder-functions=exec-count \
# RUN:   --hot-functions-at-end --align-text=4 --align-functions=4 \
# RUN:   --pad-funcs=_start:1024 --debug-only=longjmp 2>&1 | \
# RUN:   FileCheck %s --check-prefix=OLD-FAIL

# HOT: BOLT-DEBUG: LongJmp layout: section .text.cold starts at 0x{{[0-9a-f]+}}
# HOT: BOLT-DEBUG: LongJmp layout: main fragment cold_function starts at 0x[[COLD:[0-9a-f]+]]
# HOT: BOLT-DEBUG: LongJmp layout: section .text.cold ends at 0x
# HOT: BOLT-DEBUG: LongJmp layout: section .text starts at 0x{{[0-9a-f]+}}
# HOT: BOLT-DEBUG: LongJmp layout: main fragment _start starts at 0x[[START:[0-9a-f]+]]
# HOT: BOLT-DEBUG: LongJmp layout: main fragment hot starts at 0x[[HOTFUNC:[0-9a-f]+]]
# HOT: cold_function T [[COLD]]
# HOT: _start T [[START]]
# HOT: hot T [[HOTFUNC]]

# OLD: BOLT-DEBUG: LongJmp layout: section .text.cold starts at 0x{{[0-9a-f]+}}
# OLD: BOLT-DEBUG: LongJmp layout: main fragment cold_function starts at 0x[[OLD_COLD:[0-9a-f]+]]
# OLD: BOLT-DEBUG: LongJmp layout: section .text.cold ends at 0x
# OLD: BOLT-DEBUG: LongJmp layout: section .text starts at 0x{{[0-9a-f]+}}
# OLD: BOLT-DEBUG: LongJmp layout: main fragment _start starts at 0x[[OLD_START:[0-9a-f]+]]
# OLD: BOLT-DEBUG: LongJmp layout: main fragment hot starts at 0x[[OLD_HOT:[0-9a-f]+]]
# OLD: BOLT-INFO: using original .text for new code
# OLD: cold_function T [[OLD_COLD]]
# OLD: _start T [[OLD_START]]
# OLD: hot T [[OLD_HOT]]

# OLD-FORWARD: BOLT-DEBUG: LongJmp layout: section .text starts at 0x
# OLD-FORWARD: BOLT-DEBUG: LongJmp layout: main fragment _start starts at 0x[[FORWARD_START:[0-9a-f]+]]
# OLD-FORWARD: BOLT-DEBUG: LongJmp layout: main fragment hot starts at 0x[[FORWARD_HOT:[0-9a-f]+]]
# OLD-FORWARD: BOLT-DEBUG: LongJmp layout: section .text.cold starts at 0x
# OLD-FORWARD: BOLT-DEBUG: LongJmp layout: main fragment cold_function starts at 0x[[FORWARD_COLD:[0-9a-f]+]]
# OLD-FORWARD: BOLT-INFO: using original .text for new code
# OLD-FORWARD: _start T [[FORWARD_START]]
# OLD-FORWARD: hot T [[FORWARD_HOT]]
# OLD-FORWARD: cold_function T [[FORWARD_COLD]]

# OLD-FORWARD-FAIL: BOLT-WARNING: --use-old-text failed during LongJmp layout.
# OLD-FORWARD-FAIL: BOLT-WARNING: --use-old-text failed. The original .text
# OLD-FORWARD-FAIL-NOT: BOLT-INFO: using original .text for new code

# OLD-FAIL: BOLT-WARNING: --use-old-text failed during LongJmp layout.
# OLD-FAIL: BOLT-WARNING: --use-old-text failed. The original .text
# OLD-FAIL-NOT: BOLT-INFO: using original .text for new code

  .text
  .space 512, 0

  .globl _start
  .type _start, %function
_start:
.entry_start:
  bl hot
  ret
  .size _start, .-_start

  .globl hot
  .type hot, %function
hot:
.entry_hot:
  ret
  .size hot, .-hot

  .globl cold_function
  .type cold_function, %function
cold_function:
  ret
  .size cold_function, .-cold_function

  .reloc 0, R_AARCH64_NONE

# FDATA: 1 _start #.entry_start# 10
# FDATA: 1 hot #.entry_hot# 10
