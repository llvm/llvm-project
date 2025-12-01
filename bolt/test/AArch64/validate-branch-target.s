## Test that BOLT errs when detecting the target 
## of a direct call/branch is a invalid instruction

# REQUIRES: system-linux
# RUN: rm -rf %t && mkdir -p %t && cd %t
# RUN: llvm-mc -filetype=obj -triple aarch64-unknown-linux %s -o main.o
# RUN: %clang %cflags %t/main.o -o main.exe -Wl,-q
# RUN: llvm-bolt %t/main.exe -o %t/main.exe.bolt -lite=0 2>&1 | FileCheck %s --check-prefix=CHECK-TARGETS

# CHECK-TARGETS: BOLT-WARNING: corrupted control flow detected in function external_corrupt, an external branch/call targets an invalid instruction at address 0x{{[0-9a-f]+}}
# CHECK-TARGETS: BOLT-WARNING: corrupted control flow detected in function internal_corrupt, an internal branch/call targets an invalid instruction at address 0x{{[0-9a-f]+}}


.globl  internal_corrupt
.type   internal_corrupt,@function
internal_corrupt:
    ret
    nop
.Lfake_branch_1:
    .inst 0x14000001  // Opcode 0x14=b, check for internal branch: b + 0x4
.Lgarbage_1:
    .word 0xffffffff
.size   internal_corrupt,.-internal_corrupt


.globl  external_corrupt
.type   external_corrupt,@function
external_corrupt:
    ret
    nop
.Lfake_branch_2:
    .inst   0x14000004  // Opcode 0x14=b, check for external branch: b + 0xf
.size   external_corrupt,.-external_corrupt
