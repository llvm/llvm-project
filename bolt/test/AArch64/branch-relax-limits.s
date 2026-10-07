## Ensure that BOLT will error if branches are relaxed by more than 128MiB
## by the LongJmp pass.

# RUN: %clang %cflags %s -o %t -Wl,-Ttext=0x10000000
# RUN: not llvm-bolt %t -o %t.f --split-functions --split-strategy=all \
# RUN:   --custom-allocation-vma=0x20000000 2>&1 | FileCheck %s
# RUN: not llvm-bolt %t -o %t.b --split-functions --split-strategy=all \
# RUN:   --custom-allocation-vma=0x200000 2>&1 | FileCheck %s

# CHECK: BOLT-ERROR: Unable to relax non-call branch beyond 128MiB

    .text
    .globl _start
    .type _start, @function
_start:
    mov x0, #1
    b .Lcold
.Lcold:
    ret
    .size _start, .-_start
