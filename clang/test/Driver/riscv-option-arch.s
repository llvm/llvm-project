# REQUIRES: riscv-registered-target
# RUN: %clang --target=riscv64 -menable-experimental-extensions -c -o /dev/null %s
# RUN: ! %clang --target=riscv64 -c -o /dev/null %s 2>&1 | FileCheck -check-prefixes=CHECK-ERR %s

.option arch, +zicfiss
# CHECK-ERR: Unexpected experimental extensions.
