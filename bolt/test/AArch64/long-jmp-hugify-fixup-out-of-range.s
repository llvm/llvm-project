# Check that branches considered in-range during longjump
# may go out of range at JITLink if hugify moves hot code.

# REQUIRES: system-linux, asserts

# RUN: %clang %cflags -Wl,-q %s -o %t
# RUN: link_fdata --no-lbr %s %t %t.fdata
# RUN: llvm-strip --strip-unneeded %t
# RUN: not llvm-bolt %t -o %t.bolt --data %t.fdata -split-functions --hugify 2>&1 \
# RUN:   | FileCheck %s

  .globl foo
  .type foo, %function
foo:
.entry_foo:
# FDATA: 1 foo #.entry_foo# 10
    cbz x0, .Lcold_foo
    mov x0, #1
.Lcold_foo:
    ret

  .globl main
  .type main, %function
main:
    mov w0, wzr
    ret

  .globl _start
  .type _start, %function
_start:
    bl main
    b .

## Force relocation mode.
.reloc 0, R_AARCH64_NONE

# CHECK: BOLT-ERROR: JITLink failed: In graph in-memory object file, section .text: relocation target {{0x[0-9a-f]+}} {{.*}} is out of range of CondBranch19PCRel fixup at address {{0x[0-9a-f]+}} {{.*}}
