## Test that BOLT errs when detecting the target 
## of a direct call/branch is a invalid instruction

# REQUIRES: system-linux
# RUN: rm -rf %t && mkdir -p %t && cd %t
# RUN: llvm-mc -filetype=obj -triple x86_64-unknown-linux %s -o main.o
# RUN: %clang %cflags -pie -Wl,-q %t/main.o -o main.exe
# RUN: llvm-bolt %t/main.exe -o %t/main.exe.bolt -lite=0 2>&1 | FileCheck %s --check-prefix=CHECK-TARGETS

# CHECK-TARGETS: BOLT-WARNING: corrupted control flow detected in function external_corrupt: an external branch/call targets an invalid instruction in function external_func at address 0x{{[0-9a-f]+}}; ignoring both functions
# CHECK-TARGETS: BOLT-WARNING: corrupted control flow detected in function internal_corrupt: an internal branch/call targets an invalid instruction at address 0x{{[0-9a-f]+}}; ignoring this function


.globl	internal_corrupt
.type	internal_corrupt,@function
internal_corrupt:
	jb  data_in_code + 1  # targeting the data in code, and jump into the middle of 'xorb' instruction
data_in_code:
	.byte 0x34, 0x01 # data in code, will be disassembled as 'xorb 0x1, %al'
.size	internal_corrupt,.-internal_corrupt


.globl	external_corrupt
.type	external_corrupt,@function
external_corrupt:
	jb  external_func + 1  # targeting the middle of normal instruction externally
.size	external_corrupt,.-external_corrupt

.globl	external_func
.type	external_func,@function
external_func:
	addq  $1, %rax  # normal instruction
.size	external_func,.-external_func
