## Check that entry patches reflect the final function states after reference scanning.

# RUN: llvm-mc -triple=aarch64 -filetype=obj %s -o %t.o
# RUN: ld.lld --emit-relocs -e _start %t.o -o %t.exe
# RUN: llvm-bolt %t.exe -o %t.bolt --skip-funcs=_start --lite=0 --use-old-text=0
# RUN: llvm-objdump -d --section=.bolt.org.text %t.bolt | FileCheck %s

## Check that ignored_target is retained.
# CHECK-LABEL: <ignored_target{{(\.org\.0)?}}>:
# CHECK-NEXT: mov w0,

## Ensure that patched_target has the complete trampoline.
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

## No alignment or padding before _start: source has only 8 bytes available.
  .globl source
  .type source, %function
source:
  adr x8, ignored_target
  b patched_target
  .size source, .-source

  .globl _start
  .type _start, %function
_start:
  bl ignored_target
  bl source
  mov w8, #93
  svc #0
  .size _start, .-_start
