# This test checks that splitting functions which contain short range
# conditional branches works in compact code model without relying on
# relocations.

# REQUIRES: system-linux, asserts

# RUN: %clang %cflags -march=armv9-a+cmpbr -Wl,-q %s -o %t -DNUM_NOPS=0 -DRESERVE_SPACE=0
# RUN: link_fdata --no-lbr %s %t %t.fdata
# RUN: llvm-bolt %t -o %t.bolt --data %t.fdata -split-functions --compact-code-model
# RUN: llvm-objdump -d %t.bolt | FileCheck %s

# RUN: %clang %cflags -march=armv9-a+cmpbr -Wl,-q %s -o %t -DNUM_NOPS=256 -DRESERVE_SPACE=0
# RUN: link_fdata --no-lbr %s %t %t.fdata
# RUN: llvm-bolt %t -o %t.bolt --data %t.fdata -split-functions --compact-code-model --keep-nops
# RUN: llvm-objdump -d %t.bolt | FileCheck %s

# RUN: %clang %cflags -march=armv9-a+cmpbr -Wl,-q %s -o %t -DNUM_NOPS=0 -DRESERVE_SPACE=1
# RUN: link_fdata --no-lbr %s %t %t.fdata
# RUN: not llvm-bolt %t -o %t.bolt --data %t.fdata -split-functions --compact-code-model 2>&1 \
# RUN:   | FileCheck %s --check-prefix=BEYOND-128MB

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

.if RESERVE_SPACE
.space 0x8000000
.endif

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

# BEYOND-128MB: BOLT-ERROR: JITLink failed: In graph in-memory object file, section .text: relocation target {{0x[0-9a-f]+}} (<anonymous symbol>) is out of range of Branch26PCRel fixup at address {{0x[0-9a-f]+}}
