## Check that functions marked ignored after the output function list is built
## do not affect LongJmp's section layout. Use --force-patch to
## make this state deterministic. Patching late_ignored fails because the
## function is too small, so PatchEntries marks it ignored after PopulateOutputFunctions
## has already included it as a profiled function in the output list.
##
## Before the patch, the frontier scan counts late_ignored, but the subsequent
## layout loop skips it. LongJmp therefore estimates this layout:
##
##   _start.hot | 128 MiB padding | separator | _start.cold | end
##
## The conditional branch from _start.hot to _start.cold appears out of range,
## so LongJmp inserts a hot stub. The cold call to end appears to be in range,
## so no cold stub is inserted. The emitter skips late_ignored consistently
## while placing the fragments, and produces this layout instead:
##
##   _start.hot | _start.cold | 128 MiB padding | separator | end
##
## Here, the conditional branch is in range but the cold call is out of range,
## causing an emission failure. After the patch, the layout is built
## from emitted fragments grouped by output section, so late_ignored is excluded.
## The estimate then matches the emitted layout, and LongJmp inserts the required
## cold stub instead of the unnecessary hot stub.

# REQUIRES: system-linux

# RUN: llvm-mc -filetype=obj -triple aarch64-unknown-unknown %s -o %t.o
# RUN: %clang %cflags %t.o -o %t.exe -nostdlib -Wl,-q
# RUN: link_fdata --no-lbr %s %t.exe %t.fdata
# RUN: llvm-strip --strip-unneeded %t.exe
# RUN: llvm-bolt %t.exe -o %t.bolt --data %t.fdata --lite=0 \
# RUN:   --reorder-functions=exec-count --split-functions --split-all-cold \
# RUN:   --skip-funcs=skipped --force-patch \
# RUN:   --pad-funcs-before=separator:134217728 2>&1 \
# RUN:   | FileCheck %s

# CHECK: BOLT-WARNING: failed to patch entries in late_ignored
# CHECK: BOLT-INFO: Inserted 0 stubs in the hot area and 1 stubs in the cold area.

  .text
  .globl _start
  .type _start, %function
_start:
.entry_start:
# FDATA: 1 _start #.entry_start# 10
  bl late_ignored
  cbnz x0, .hot_start
.cold_start:
  bl end
  ret
.hot_start:
# FDATA: 1 _start #.hot_start# 10
  ret
  .size _start, .-_start

  .globl late_ignored
  .type late_ignored, %function
late_ignored:
.entry_late_ignored:
# FDATA: 1 late_ignored #.entry_late_ignored# 5
  ret
  .size late_ignored, .-late_ignored

  .globl skipped
  .type skipped, %function
skipped:
  ret
  .size skipped, .-skipped

  .globl separator
  .type separator, %function
separator:
  add x0, x0, #1
  add x0, x0, #1
  ret
  .size separator, .-separator

  .globl end
  .type end, %function
end:
  add x1, x1, #1
  add x1, x1, #1
  ret
  .size end, .-end
