## Test that BOLT errs when detecting the target 
## of a direct call/branch is a invalid instruction

# REQUIRES: system-linux
# RUN: rm -rf %t && mkdir -p %t && cd %t
# RUN: llvm-mc -filetype=obj -triple aarch64-unknown-linux %s -o main.o
# RUN: %clang %cflags %t/main.o -o main.exe -Wl,-q
# RUN: llvm-bolt %t/main.exe -o %t/main.exe.bolt -lite=0 2>&1 | FileCheck %s --check-prefix=CHECK-TARGETS

# CHECK-TARGETS: BOLT-WARNING: corrupted control flow detected in function external_corrupt: an external branch/call targets an invalid instruction in function external_func at address 0x{{[0-9a-f]+}}; ignoring both functions
# CHECK-TARGETS: BOLT-WARNING: corrupted control flow detected in function internal_corrupt: an internal branch/call targets an invalid instruction at address 0x{{[0-9a-f]+}}; ignoring this function


.globl  internal_corrupt
.type   internal_corrupt,@function
internal_corrupt:
    b constant_island_0  // targeting the data in code
constant_island_0:
    .word 0xffffffff
.size   internal_corrupt,.-internal_corrupt


.globl  external_corrupt
.type   external_corrupt,@function
external_corrupt:
    b   constant_island_1  // targeting the data in code externally
.size   external_corrupt,.-external_corrupt

.globl  external_func
.type   external_func,@function
external_func:
    add x0, x0, x1
constant_island_1:
    .word 0xffffffff // data in code
    ret
.size   external_func,.-external_func
