## Check that HLT instruction is handled differently depending on the flags.
## It's a terminator in the user-level code, but the execution can resume in
## ring 0.

# RUN: %clang %cflags %s -static -o %t.exe -nostdlib
# RUN: llvm-bolt %t.exe --print-cfg --print-only=main --terminal-x86-hlt=0 \
# RUN:   -o %t.ring0 2>&1 | FileCheck %s --check-prefix=CHECK-RING0
# RUN: llvm-bolt %t.exe --print-cfg --print-only=main \
# RUN:   -o %t.ring3 2>&1 | FileCheck %s --check-prefix=CHECK-RING3
# RUN: llvm-objdump -d %t.ring0 --print-imm-hex | FileCheck %s --check-prefix=CHECK-BIN

# CHECK-RING0: BB Count    : 1
# CHECK-RING3: BB Count    : 2

# CHECK-BIN: <main>:
# CHECK-BIN-NEXT: f4                            hlt
# CHECK-BIN-NEXT: c3                            retq

.global main
  .type main, %function
main:
        hlt
        retq
.size main, .-main
