## Check that LongJmp uses the pre-assigned address of an injected patch.

# REQUIRES: system-linux, asserts

# RUN: llvm-mc -filetype=obj -triple aarch64-unknown-unknown %s -o %t.o
# RUN: %clang %cflags %t.o -o %t.exe -nostdlib -Wl,-q
# RUN: llvm-bolt %t.exe -o %t.bolt --lite=0 --use-old-text=0 \
# RUN:   --force-patch --debug-only=longjmp > %t.log 2>&1
# RUN: llvm-nm -n --format=posix %t.bolt >> %t.log
# RUN: FileCheck %s < %t.log

# CHECK: BOLT-DEBUG: LongJmp layout: main fragment patched.org.0/ starts at 0x[[PATCH:[0-9a-f]+]]
# CHECK: patched.org.0 t [[PATCH]]

  .text
  .balign 4
  .globl patched
  .type patched, %function
patched:
  .rept 32
  nop
  .endr
  ret
  .size patched, .-patched

  .globl _start
  .type _start, %function
_start:
  bl patched
  ret
  .size _start, .-_start
