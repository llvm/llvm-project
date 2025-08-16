## Check CFG for halt instruction

# RUN: %clang %cflags %s -static -o %t.exe -nostdlib
# RUN: llvm-bolt %t.exe --print-cfg --print-only=main -o %t 2>&1 | FileCheck %s --check-prefix=CHECK-CFG
# RUN: llvm-objdump -d %t --print-imm-hex | FileCheck %s --check-prefix=CHECK-BIN

# CHECK-CFG: BB Count    : 1
# CHECK-BIN: <main>:
# CHECK-BIN-NEXT: f4                            hlt
# CHECK-BIN-NEXT: c3                            retq

.global main
  .type main, %function
main:
        hlt
        retq
.size main, .-main
