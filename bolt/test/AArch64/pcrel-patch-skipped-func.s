## Check that BOLT does not create an unencodable instruction patch for a
## short-range PC-relative reference in a function it was asked to skip. Here
## the reference is an ADR; see pcrel-patch-ldr-literal.s for the LDR (literal)
## form, which fails the same way.
##
## The ADR target is resolved by the assembler, so the binary carries no
## R_AARCH64_ADR_PREL_LO21 relocation and BOLT's symbolizer is the only source
## of the reference, which bypasses the existing ADR guard in
## scanExternalRefs().
##
## Without a fix:
##   BOLT-ERROR: JITLink failed: ... section .local.text.__BP_0: relocation
##   target 0x400014 ... is out of range of ADRLiteral21 fixup ...
##
## https://github.com/llvm/llvm-project/issues/194938

# REQUIRES: system-linux

# RUN: llvm-mc -filetype=obj -triple aarch64-unknown-unknown %s -o %t.o
# RUN: %clang %cflags %t.o -o %t.exe -nostdlib -fuse-ld=lld -Wl,-q
# RUN: llvm-bolt %t.exe -o %t.bolt --skip-funcs=skip_me 2>&1 \
# RUN:   | FileCheck %s --check-prefix=CHECK-BOLT
# RUN: llvm-objdump -d --disassemble-symbols=skip_me %t.bolt | FileCheck %s

# CHECK-BOLT: BOLT-WARNING: unable to update PC-relative reference to
# CHECK-BOLT-NOT: BOLT-ERROR

## The skipped function is preserved in .bolt.org.text and its ADR is left
## alone, still reaching the data at its original address.
# CHECK-LABEL: <skip_me>:
# CHECK: adr x8

  .text
  .globl _start
  .type _start, %function
_start:
  mov w0, #1
  bl skip_me
  bl hot_func
  mov w8, #93
  svc #0
  .size _start, .-_start

## Data in .text, referenced by skip_me via ADR. BOLT registers this untyped
## local label as a function holding a constant island and emits it into the new
## text, which is where the ADR reference would have to point.
  .p2align 2
my_data:
  .word 0x10, 0x20, 0x30, 0x40

  .globl skip_me
  .type skip_me, %function
skip_me:
  cmp w0, #3
  b.hi .Lbad
  adr x8, my_data
  ldr w0, [x8, w0, uxtw #2]
  ret
.Lbad:
  mov w0, #-1
  ret
  .size skip_me, .-skip_me

  .globl hot_func
  .type hot_func, %function
hot_func:
  mul w0, w0, w0
  add w0, w0, #1
  ret
  .size hot_func, .-hot_func
