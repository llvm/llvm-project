# This test checks that the non-compact LongJmp path inserts a stub when a
# compare-and-branch target is moved beyond its +/-1KB range.

# REQUIRES: system-linux

# RUN: %clang %cflags -march=armv9-a+cmpbr -Wl,-q %s -o %t
# RUN: link_fdata --no-lbr %s %t %t.fdata
# RUN: llvm-strip --strip-unneeded %t
# RUN: llvm-bolt %t -o %t.bolt --data %t.fdata --split-functions --keep-nops
# RUN: llvm-objdump --no-show-raw-insn -d %t.bolt | FileCheck %s

  .globl foo
  .type foo, %function
foo:
.entry_foo:
# FDATA: 1 foo #.entry_foo# 10
    cbgt x0, #0, .Lcold_foo
    mov x0, #1
.Lcold_foo:
    ret

  .globl large_function
  .type large_function, %function
large_function:
.entry_large_function:
# FDATA: 1 large_function #.entry_large_function# 10
  .rept 256
    nop
  .endr
    ret

## Force relocation mode.
.reloc 0, R_AARCH64_NONE

# CHECK: Disassembly of section .text:

# CHECK: <foo>:
# CHECK-NEXT:           {{.*}} cblt x0, #0x1, 0x[[STUB:[0-9a-f]+]] <{{.*}}>
# CHECK-NEXT:           {{.*}} b              0x[[TARGET:[0-9a-f]+]] <{{.*}}>
# CHECK-NEXT: [[STUB]]: b 0x[[FALLTHROUGH:[0-9a-f]+]] <{{.*}}>

# CHECK: Disassembly of section .text.cold:

# CHECK: <foo.cold.0>:
# CHECK-NEXT: [[FALLTHROUGH]]: mov x0, #0x1
# CHECK-NEXT: [[TARGET]]: ret
