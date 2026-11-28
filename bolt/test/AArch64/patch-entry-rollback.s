## Check that ignoring a function removes its previously registered entry patch.

# RUN: llvm-mc -triple=aarch64 -filetype=obj %s -o %t.o
# RUN: ld.lld --emit-relocs -e _start %t.o -o %t.exe
# RUN: llvm-bolt %t.exe -o %t.bolt --skip-funcs=_start --lite=0 --use-old-text=0
# RUN: llvm-objdump -d --section=.bolt.org.text %t.bolt | FileCheck %s

# CHECK-LABEL: <ignored_target>:
# CHECK-NEXT: mov w0, #{{(0x2a|42)}}
# CHECK-NEXT: nop
# CHECK-NEXT: ret

# CHECK-LABEL: <patched_target.org.0>:
# CHECK-NEXT: adrp x16,
# CHECK-NEXT: add x16, x16,
# CHECK-NEXT: br x16

  .text
  .p2align 2
  .globl ignored_target
  .type ignored_target, %function
ignored_target:
  mov w0, #42
  nop
  ret
  .size ignored_target, .-ignored_target

  .globl patched_target
  .type patched_target, %function
patched_target:
  mov w0, #42
  nop
  ret
  .size patched_target, .-patched_target

## This function is too short for a 12-byte trampoline. Scanning its original
## instructions after ignoring it forces ignored_target to stay at its address.
  .globl source
  .type source, %function
source:
  adr x8, ignored_target
  ret
  .size source, .-source

  .globl _start
  .type _start, %function
_start:
  bl ignored_target
  bl patched_target
  bl source
  mov w8, #93
  svc #0
  .size _start, .-_start
