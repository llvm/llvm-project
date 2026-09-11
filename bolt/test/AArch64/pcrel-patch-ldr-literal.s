## Same failure class as pcrel-patch-skipped-func.s, but through LDR (literal)
## instead of ADR. LDR (literal) has the same +/-1MB reach and the same
## assembler-resolution behaviour for a local symbol in the same section, so it
## produces an unencodable patch with an LDRLiteral19 fixup:
##
##   BOLT-ERROR: JITLink failed: ... section .local.text.__BP_0: relocation
##   target 0x400000 ... is out of range of LDRLiteral19 fixup ...
##
## A fix that only looks at ADR does not cover this.
##
## https://github.com/llvm/llvm-project/issues/194938

# REQUIRES: system-linux

# RUN: llvm-mc -filetype=obj -triple aarch64-unknown-unknown %s -o %t.o
# RUN: %clang %cflags %t.o -o %t.exe -nostdlib -fuse-ld=lld -Wl,-q
## The only relocation in the binary is the CALL26 for the global bl target:
## the LDR (literal) below was resolved by the assembler.
# RUN: llvm-readobj -r %t.exe | FileCheck %s --check-prefix=CHECK-RELOC
# RUN: llvm-bolt %t.exe -o %t.bolt --funcs='target_fn.*' 2>&1 \
# RUN:   | FileCheck %s --check-prefix=CHECK-BOLT
# RUN: llvm-objdump -d --disassemble-symbols=victim %t.bolt | FileCheck %s

# CHECK-RELOC: R_AARCH64_CALL26 victim
# CHECK-RELOC-NOT: R_AARCH64_LD_PREL_LO19

# CHECK-BOLT: BOLT-WARNING: unable to update PC-relative reference to
# CHECK-BOLT-NOT: BOLT-ERROR

## The target is kept in place, so the reference stays valid and the function
## address is unchanged.
# RUN: llvm-nm %t.exe | grep -w target_fn > %t.orig.sym
# RUN: llvm-nm %t.bolt | grep -w target_fn > %t.new.sym
# RUN: diff %t.orig.sym %t.new.sym

# CHECK-LABEL: <victim>:
# CHECK: ldr x8

  .text
  .globl _start
  .type _start, %function
_start:
  mov w0, #5
  bl target_fn
  bl victim
  mov w8, #93
  svc #0
  .size _start, .-_start

## Local symbol: the LDR (literal) below carries no relocation.
  .type target_fn, %function
target_fn:
  mul w0, w0, w0
  add w0, w0, #1
  ret
  .size target_fn, .-target_fn

  .globl victim
  .type victim, %function
victim:
  ldr x8, target_fn
  ret
  .size victim, .-victim
