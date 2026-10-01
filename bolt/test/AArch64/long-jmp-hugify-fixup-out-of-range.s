# The longjump pass may consider branch targets in range during tentative
# layout and decide not to insert stubs for them. Later, final section
# allocation may insert alignment padding after the last non-cold text section
# when hugify is enabled. This moves the following cold section farther away,
# resulting in relocation fixups going out of range at JITLink. Check that the
# longjump pass accounts for this padding and inserts stubs when needed.

# REQUIRES: system-linux, asserts, bolt-runtime, target=aarch64{{.*}}

# RUN: %clang %cflags -Wl,-q %s -o %t
# RUN: link_fdata --no-lbr %s %t %t.fdata
# RUN: llvm-strip --strip-unneeded %t
# RUN: llvm-bolt %t -o %t.bolt --data %t.fdata -split-functions --hugify
# RUN: llvm-objdump -d %t.bolt | FileCheck %s

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

# CHECK: Disassembly of section .text:

# CHECK: <foo>:
# CHECK-NEXT:            {{.*}} cbnz x0, 0x[[ADDR0:[0-9a-f]+]] <{{.*}}>
# CHECK-NEXT:            {{.*}} b 0x[[ADDR1:[0-9a-f]+]] <{{.*}}>
# CHECK-NEXT: [[ADDR0]]: {{.*}} b 0x[[ADDR2:[0-9a-f]+]] <{{.*}}>

# CHECK: Disassembly of section .text.cold:

# CHECK: <foo.cold.0>:
# CHECK-NEXT: [[ADDR2]]: {{.*}} mov x0, #0x1 // =1
# CHECK-NEXT: [[ADDR1]]: {{.*}} ret
