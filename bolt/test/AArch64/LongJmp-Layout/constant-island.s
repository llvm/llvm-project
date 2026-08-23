## Check layout of an aligned constant island. With splitting, the
## cold block's reference also exercises duplicated constant-island emission.

# REQUIRES: system-linux, asserts

# RUN: llvm-mc -filetype=obj -triple aarch64-unknown-unknown %s -o %t.o
# RUN: %clang %cflags %t.o -o %t.exe -nostdlib -Wl,-q
# RUN: link_fdata --no-lbr %s %t.exe %t.fdata
# RUN: llvm-bolt %t.exe -o %t.unsplit.bolt --data %t.fdata --lite=0 \
# RUN:   --debug-only=longjmp 2>&1 | FileCheck %s
# RUN: llvm-readelf -S %t.unsplit.bolt | FileCheck %s \
# RUN:   --check-prefix=UNSPLIT-SECTIONS \
# RUN:   --implicit-check-not='.text.cold'
# RUN: llvm-bolt %t.exe -o %t.split.bolt --data %t.fdata --lite=0 \
# RUN:   --split-functions --split-all-cold --debug-only=longjmp 2>&1 \
# RUN:   | FileCheck %s
# RUN: llvm-readelf -S %t.split.bolt | FileCheck %s \
# RUN:   --check-prefix=SPLIT-SECTIONS

# CHECK: BOLT-DEBUG: LongJmp: layout iteration 1
# CHECK: BOLT-DEBUG: LongJmp layout starts at 0x
# CHECK: BOLT-DEBUG: LongJmp layout: section .text starts at 0x
# CHECK: BOLT-DEBUG: LongJmp layout: main fragment
# CHECK: BOLT-DEBUG: LongJmp layout: basic block
# CHECK: BOLT-DEBUG: LongJmp layout: section .text ends at 0x

# UNSPLIT-SECTIONS: .text
# SPLIT-SECTIONS: .text
# SPLIT-SECTIONS: .text.cold

  .text
  .globl _start
  .type _start, %function
_start:
.entry_start:
# FDATA: 1 _start #.entry_start# 10
  bl island_user
  ret
  .size _start, .-_start

  .globl island_user
  .type island_user, %function
island_user:
.entry_island_user:
# FDATA: 1 island_user #.entry_island_user# 10
  cbz x0, .Lcold
.hot_island_user:
# FDATA: 1 island_user #.hot_island_user# 10
  ret
.Lcold:
  adr x1, .Lconstant_island
  ldr x1, [x1]
  ret
  .size island_user, .-island_user

  .p2align 6
.Lconstant_island:
  .xword 0x1122334455667788
  .xword 0x8877665544332211

  .reloc 0, R_AARCH64_NONE
