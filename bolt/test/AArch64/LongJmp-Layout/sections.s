## Check section-aware layout for profiled functions in .text,
## unprofiled whole functions in .text.cold, and cold basic blocks split from a
## hot function. The middle cold block is split only with --split-all-cold.

# REQUIRES: system-linux, asserts

# RUN: split-file %s %t
# RUN: llvm-mc -filetype=obj -triple aarch64-unknown-unknown %t/input.s -o %t.o
# RUN: %clang %cflags %t.o -o %t.exe -nostdlib -Wl,-q
# RUN: llvm-bolt %t.exe -o %t.unsplit.bolt --data %t/profile.fdata --lite=0 \
# RUN:   --reorder-functions=exec-count --debug-only=longjmp 2>&1 \
# RUN:   | FileCheck %s
# RUN: llvm-nm -n %t.unsplit.bolt | FileCheck %s --check-prefix=UNSPLIT \
# RUN:   --implicit-check-not='hot.cold.0' \
# RUN:   --implicit-check-not='trailing.cold.0'
# RUN: llvm-bolt %t.exe -o %t.split.bolt --data %t/profile.fdata --lite=0 \
# RUN:   --reorder-functions=exec-count --split-functions \
# RUN:   --debug-only=longjmp 2>&1 | FileCheck %s
# RUN: llvm-nm -n %t.split.bolt | FileCheck %s --check-prefix=SPLIT \
# RUN:   --implicit-check-not='hot.cold.0'
# RUN: llvm-bolt %t.exe -o %t.split-all.bolt --data %t/profile.fdata --lite=0 \
# RUN:   --reorder-functions=exec-count --split-functions --split-all-cold \
# RUN:   --debug-only=longjmp 2>&1 | FileCheck %s
# RUN: llvm-nm -n %t.split-all.bolt | FileCheck %s --check-prefix=SPLIT-ALL

# CHECK: BOLT-DEBUG: LongJmp: layout iteration 1
# CHECK: BOLT-DEBUG: LongJmp layout starts at 0x
# CHECK: BOLT-DEBUG: LongJmp layout: section .text starts at 0x
# CHECK: BOLT-DEBUG: LongJmp layout: main fragment
# CHECK: BOLT-DEBUG: LongJmp layout: basic block
# CHECK: BOLT-DEBUG: LongJmp layout: section .text ends at 0x

# UNSPLIT: T hot
# UNSPLIT: T trailing
# SPLIT: t trailing.cold.0
# SPLIT-ALL: t hot.cold.0
# SPLIT-ALL: t trailing.cold.0

#--- input.s
  .text
  .globl _start
  .type _start, %function
_start:
  bl hot
  bl trailing
  ret
  .size _start, .-_start

  .globl hot
  .type hot, %function
hot:
  cbnz x0, .Lhot
.Lcold_middle:
  sub x0, x0, #1
  b .Lexit
.Lhot:
  add x0, x0, #1
.Lexit:
  ret
  .size hot, .-hot

  .globl trailing
  .type trailing, %function
trailing:
  cbz x1, .Ltrailing_cold
  add x1, x1, #1
  ret
.Ltrailing_cold:
  sub x1, x1, #1
  ret
  .size trailing, .-trailing

  .globl cold_function
  .type cold_function, %function
cold_function:
  add x2, x2, #1
  ret
  .size cold_function, .-cold_function

  .reloc 0, R_AARCH64_NONE

#--- profile.fdata
no_lbr
1 _start 0 10
1 hot 0 10
1 hot 4 0
1 hot c 10
1 hot 10 10
1 trailing 0 10
1 trailing 4 10
1 trailing c 0
