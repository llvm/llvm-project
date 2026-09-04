## Check that a skipped function does not contribute to the layout.
## Padding puts end just within range of _start's call. The intervening target
## function pushes end out of range unless target is skipped.
##
## 134217716 is 0x7fffff4. With target skipped, the displacement is:
##
##   0x7fffff4 (padding) + 8 (_start) = 0x7fffffc
##
## This is the largest positive displacement encodable by bl. When target is
## emitted, its 8-byte size increases the displacement to 0x8000004, requiring
## a stub.

# REQUIRES: system-linux

# RUN: llvm-mc -filetype=obj -triple aarch64-unknown-unknown %s -o %t.o
# RUN: %clang %cflags %t.o -o %t.exe -nostdlib -Wl,-q
# RUN: llvm-strip --strip-unneeded %t.exe
# RUN: llvm-bolt %t.exe -o %t.relaxed.bolt --lite=0 \
# RUN:   --pad-funcs-before=end:134217716 2>&1 \
# RUN:   | FileCheck %s --check-prefix=RELAXED
# RUN: llvm-objdump -d --no-show-raw-insn %t.relaxed.bolt \
# RUN:   | FileCheck %s --check-prefix=STUB-DISASM
# RUN: llvm-bolt %t.exe -o %t.skip.bolt --lite=0 \
# RUN:   --pad-funcs-before=end:134217716 --skip-funcs=target 2>&1 \
# RUN:   | FileCheck %s --check-prefix=SKIP
# RUN: llvm-objdump -d --no-show-raw-insn %t.skip.bolt \
# RUN:   | FileCheck %s --check-prefix=DIRECT-DISASM

# RELAXED: BOLT-INFO: Inserted 1 stubs in the hot area and 0 stubs in the cold area.
# SKIP: BOLT-INFO: Inserted 0 stubs in the hot area and 0 stubs in the cold area.

# STUB-DISASM-LABEL: <_start>:
# STUB-DISASM-NEXT: {{.*}} bl {{.*}} <_start+0x8>
# STUB-DISASM-NEXT: {{.*}} ret
# STUB-DISASM-NEXT: {{.*}} adrp x16,
# STUB-DISASM-NEXT: {{.*}} add x16, x16,
# STUB-DISASM-NEXT: {{.*}} br x16
# DIRECT-DISASM-LABEL: <_start>:
# DIRECT-DISASM-NEXT: {{.*}} bl {{.*}} <end>
# DIRECT-DISASM-NEXT: {{.*}} ret

  .text
  .globl _start
  .type _start, %function
_start:
  bl end
  ret
  .size _start, .-_start

  .globl target
  .type target, %function
target:
  add x0, x0, #1
  ret
  .size target, .-target

  .globl end
  .type end, %function
end:
  ret
  .size end, .-end
