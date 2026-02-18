# This test checks that splitting functions which contain short range
# conditional branches works without compact code model.

# REQUIRES: system-linux, asserts

# RUN: %clang %cflags -march=armv9-a+cmpbr -Wl,-q %s -o %t -DNUM_NOPS=0 -DRESERVE_SPACE=0
# RUN: link_fdata --no-lbr %s %t %t.fdata
# RUN: llvm-bolt %t -o %t.bolt --data %t.fdata -split-functions
# RUN: llvm-objdump -d %t.bolt | FileCheck %s

# RUN: %clang %cflags -march=armv9-a+cmpbr -Wl,-q %s -o %t -DNUM_NOPS=256 -DRESERVE_SPACE=0
# RUN: link_fdata --no-lbr %s %t %t.fdata
# RUN: llvm-bolt %t -o %t.bolt --data %t.fdata -split-functions --keep-nops
# RUN: llvm-objdump -d %t.bolt | FileCheck %s

# RUN: %clang %cflags -march=armv9-a+cmpbr -Wl,-q %s -o %t -DNUM_NOPS=0 -DRESERVE_SPACE=1
# RUN: link_fdata --no-lbr %s %t %t.fdata
# RUN: llvm-bolt %t -o %t.bolt --data %t.fdata -split-functions
# RUN: llvm-objdump -d %t.bolt | FileCheck %s --check-prefix=BEYOND-128MB

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
# CHECK-NEXT:            {{.*}} cblt x0, #0x1, 0x[[ADDR0:[0-9a-f]+]] <{{.*}}>
# CHECK-NEXT:            {{.*}} b              0x[[ADDR1:[0-9a-f]+]] <{{.*}}>
# CHECK-NEXT: [[ADDR0]]: {{.*}} b              0x[[ADDR2:[0-9a-f]+]] <{{.*}}>

# CHECK: Disassembly of section .text.cold:

# CHECK: <foo.cold.0>:
# CHECK-NEXT: [[ADDR2]]: {{.*}} mov x0, #0x1 // =1
# CHECK-NEXT: [[ADDR1]]: {{.*}} ret

# BEYOND-128MB: Disassembly of section .text:

## FIXME: x16 may be clobbered here!
# BEYOND-128MB: <foo>:
# BEYOND-128MB-NEXT:            {{.*}} cbgt x0, #0x0,  0x[[ADDR0:[0-9a-f]+]]   <{{.*}}>
# BEYOND-128MB-NEXT:            {{.*}} adrp x16,       0x[[ADDR1:[0-9a-f]+]]00 <{{.*}}>
# BEYOND-128MB-NEXT:            {{.*}} add  x16, x16, #0x40
# BEYOND-128MB-NEXT:            {{.*}} br   x16
# BEYOND-128MB-NEXT: [[ADDR0]]: {{.*}} adrp x16,       0x[[ADDR1]]00 <{{.*}}>
# BEYOND-128MB-NEXT:            {{.*}} add  x16, x16, #0x44
# BEYOND-128MB-NEXT:            {{.*}} br   x16

# BEYOND-128MB: Disassembly of section .text.cold:

# BEYOND-128MB: <foo.cold.0>:
# BEYOND-128MB-NEXT: [[ADDR1]]40: {{.*}} mov x0, #0x1 // =1
# BEYOND-128MB-NEXT: [[ADDR1]]44: {{.*}} ret
