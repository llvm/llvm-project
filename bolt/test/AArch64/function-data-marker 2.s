## Check that if a data marker is present at the start of a function, the
## underlying bytes are still treated as code.

# RUN: %clang %cflags %s -o %t.exe
# RUN: llvm-bolt %t.exe -o %t.bolt --print-cfg 2>&1 | FileCheck %s

# CHECK: BOLT-WARNING: ignoring data marker conflicting with function symbol _start

.text
.balign 4

## Data marker is emitted because ".long" directive is used instead of ".inst".
.global _start
.type _start, %function
_start:
  .long 0xcec08000 // sha512su0 v0.2d, v0.2d
  ret
.size _start, .-_start

# CHECK-LABEL: Binary Function "_start"
# CHECK: Entry Point
# CHECK-NEXT: sha512su0 v0.2d, v0.2d

