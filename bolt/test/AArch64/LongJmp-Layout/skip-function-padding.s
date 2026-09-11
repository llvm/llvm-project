## Check that a skipped function and padding requested for it are both absent
## from LongJmp's emitted-section layout.

# REQUIRES: system-linux, asserts

# RUN: llvm-mc -filetype=obj -triple aarch64-unknown-unknown %s -o %t.o
# RUN: %clang %cflags %t.o -o %t.exe -nostdlib -Wl,-q
# RUN: llvm-bolt %t.exe -o %t.bolt --lite=0 --skip-funcs=skipped \
# RUN:   --pad-funcs-before=skipped:64 --pad-funcs=skipped:64 \
# RUN:   --debug-only=longjmp > %t.log 2>&1
# RUN: llvm-nm -n --format=posix %t.bolt >> %t.log
# RUN: FileCheck %s < %t.log

# CHECK: BOLT-DEBUG: LongJmp: layout iteration 1
# CHECK: BOLT-DEBUG: LongJmp layout: section .text starts at 0x
# CHECK: BOLT-DEBUG: LongJmp layout: main fragment _start starts at 0x[[START:[0-9a-f]+]]
# CHECK: BOLT-DEBUG: LongJmp layout: main fragment end starts at 0x[[END:[0-9a-f]+]]
# CHECK: BOLT-DEBUG: LongJmp layout: section .text ends at 0x
# CHECK: _start T [[START]]
# CHECK-NEXT: end T [[END]]

  .text
  .globl _start
  .type _start, %function
_start:
  bl end
  ret
  .size _start, .-_start

  .globl skipped
  .type skipped, %function
skipped:
  add x0, x0, #1
  ret
  .size skipped, .-skipped

  .globl end
  .type end, %function
end:
  add x1, x1, #1
  ret
  .size end, .-end

  .reloc 0, R_AARCH64_NONE
