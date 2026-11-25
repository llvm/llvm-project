# Check that function splitting handles short range conditional
# branches without requiring a corresponding relocation type.

# REQUIRES: system-linux, asserts

# RUN: %clang %cflags -march=armv9-a+cmpbr -Wl,-q %s -o %t -DNUM_NOPS=0
# RUN: link_fdata --no-lbr %s %t %t.fdata
# RUN: llvm-strip --strip-unneeded %t
# RUN: llvm-bolt %t -o %t.bolt --data %t.fdata -split-functions
# RUN: llvm-objdump -d %t.bolt | FileCheck %s

# RUN: %clang %cflags -march=armv9-a+cmpbr -Wl,-q %s -o %t -DNUM_NOPS=256
# RUN: link_fdata --no-lbr %s %t %t.fdata
# RUN: llvm-strip --strip-unneeded %t
# RUN: llvm-bolt %t -o %t.bolt --data %t.fdata -split-functions --keep-nops
# RUN: llvm-objdump -d %t.bolt | FileCheck %s

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
  .rept NUM_NOPS
    nop
  .endr
    ret

## Force relocation mode.
.reloc 0, R_AARCH64_NONE

# CHECK: Disassembly of section .text:

# CHECK: <foo>:
# CHECK-NEXT:            {{.*}} cbgt x0, #0x0, 0x[[ADDR0:[0-9a-f]+]] <{{.*}}>
# CHECK-NEXT:            {{.*}} b              0x[[ADDR1:[0-9a-f]+]] <{{.*}}>
# CHECK-NEXT: [[ADDR0]]: {{.*}} b              0x[[ADDR2:[0-9a-f]+]] <{{.*}}>

# CHECK: Disassembly of section .text.cold:

# CHECK: <foo.cold.0>:
# CHECK-NEXT: [[ADDR1]]: {{.*}} mov x0, #0x1 // =1
# CHECK-NEXT: [[ADDR2]]: {{.*}} ret
