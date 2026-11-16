## Check that an ADR with an R_AARCH64_ADR_PREL_LO21 relocation in a function
## that BOLT does not emit keeps its target in place. This is the
## -ffunction-sections form of pcrel-patch-skipped-func.s: victim and target_fn
## live in separate sections, so the assembler cannot resolve the ADR and emits
## a relocation for it.

# REQUIRES: system-linux

# RUN: llvm-mc -filetype=obj -triple aarch64-unknown-unknown %s -o %t.o
# RUN: llvm-readobj -r %t.o | FileCheck %s --check-prefix=CHECK-RELOC
# RUN: %clang %cflags %t.o -o %t.exe -nostdlib -fuse-ld=lld -Wl,-q
# RUN: llvm-bolt %t.exe -o %t.bolt --skip-funcs=victim 2>&1 \
# RUN:   | FileCheck %s --check-prefix=CHECK-BOLT
# RUN: llvm-objdump -d --disassemble-symbols=victim %t.bolt | FileCheck %s

# CHECK-RELOC: R_AARCH64_ADR_PREL_LO21 .text.target_fn

# CHECK-BOLT: BOLT-WARNING: unable to update PC-relative reference to target_fn
# CHECK-BOLT-NOT: BOLT-ERROR

## The target is kept in place, so the reference stays valid and the function
## address is unchanged.
# RUN: llvm-nm %t.exe | grep -w target_fn > %t.orig.sym
# RUN: llvm-nm %t.bolt | grep -w target_fn > %t.new.sym
# RUN: diff %t.orig.sym %t.new.sym

# CHECK-LABEL: <victim>:
# CHECK: adr x8

  .section .text._start,"ax",@progbits
  .globl _start
  .type _start, %function
_start:
  mov w0, #5
  bl target_fn
  bl victim
  mov w8, #93
  svc #0
  .size _start, .-_start

  .section .text.target_fn,"ax",@progbits
  .type target_fn, %function
target_fn:
  mul w0, w0, w0
  add w0, w0, #1
  ret
  .size target_fn, .-target_fn

  .section .text.victim,"ax",@progbits
  .globl victim
  .type victim, %function
victim:
  adr x8, target_fn
  blr x8
  ret
  .size victim, .-victim
