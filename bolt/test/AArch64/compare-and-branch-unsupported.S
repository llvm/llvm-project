# This test checks that splitting functions which contain short range
# conditional branches fails in compact code model due to the cold
# target being far away, beyond the 128MB limit.

# REQUIRES: system-linux, asserts

# RUN: %clang %cflags -march=armv9-a+cmpbr -Wl,-q %s -o %t
# RUN: link_fdata --no-lbr %s %t %t.fdata
# RUN: not llvm-bolt %t -o %t.bolt --data %t.fdata -split-functions --compact-code-model 2>&1 \
# RUN:   | FileCheck %s --check-prefix=COMPACT-CODE-FAILS

# RUN: %clang %cflags -march=armv9-a+cmpbr -Wl,-q %s -o %t
# RUN: link_fdata --no-lbr %s %t %t.fdata
# RUN: llvm-bolt %t -o %t.bolt --data %t.fdata -split-functions
# RUN: llvm-objdump -d %t.bolt | FileCheck %s --check-prefix=CHECK-ADRP

  .globl foo
  .type foo, %function
foo:
# FDATA: 1 foo #.entry_foo# 10
.entry_foo:
    cbgt x0, #0, .Lcold_foo
    mov x0, #1
.Lcold_foo:
    ret

## Add enough space for the cold section to be far away,
## beyond the 128MB limit of compact code model.
.space 0x8000000

## Force relocation mode.
.reloc 0, R_AARCH64_NONE


# COMPACT-CODE-FAILS: BOLT-ERROR: JITLink failed: In graph in-memory object file, section .text: relocation target {{.*}} is out of range of Branch26PCRel fixup at address {{.*}}


# CHECK-ADRP: Disassembly of section .text:

# CHECK-ADRP: <foo>:
# CHECK-ADRP-NEXT:            {{.*}} cbgt x0, #0x0, 0x[[ADDR0:[0-9a-f]+]] <{{.*}}>
# CHECK-ADRP-NEXT:            {{.*}} adrp x16,      0x[[ADDR1:[0-9a-f]+]]00 <foo+0x8000000>
# CHECK-ADRP-NEXT:            {{.*}} add  x16, x16, #0x40
# CHECK-ADRP-NEXT:            {{.*}} br   x16
# CHECK-ADRP-NEXT: [[ADDR0]]: {{.*}} adrp x16,      0x[[ADDR1]]00 <foo+0x8000000>
# CHECK-ADRP-NEXT:            {{.*}} add  x16, x16, #0x44
# CHECK-ADRP-NEXT:            {{.*}} br   x16


# CHECK-ADRP: Disassembly of section .text.cold:

# CHECK-ADRP: <foo.cold.0>:
# CHECK-ADRP-NEXT: [[ADDR1]]40: {{.*}} mov x0, #0x1 // =1
# CHECK-ADRP-NEXT: [[ADDR1]]44: {{.*}} ret
