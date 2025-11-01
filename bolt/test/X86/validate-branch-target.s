## Test that BOLT errs when detecting the target 
## of a direct call/branch is a invalid instruction

# REQUIRES: system-linux
# RUN: rm -rf %t && mkdir -p %t && cd %t
# RUN: llvm-mc -filetype=obj -triple x86_64-unknown-linux %s -o main.o
# RUN: %clang %cflags -pie -Wl,-q %t/main.o -o main.exe
# RUN: llvm-bolt %t/main.exe -o %t/main.exe.bolt 2>&1 | FileCheck %s --check-prefix=CHECK-TARGETS

# CHECK-TARGETS: BOLT-WARNING: direct branch/call at 0x{{[0-9a-f]+}} in function RC4_options targets an invalid instruction at 0x{{[0-9a-f]+}}

# a date-in-code function case from OPENSSL
.globl	RC4_options
.type	RC4_options,@function
.align	16
RC4_options:
	leaq	.Lopts(%rip),%rax
	btl	$20,%edx
	jc	.L8xchar
	btl	$30,%edx
	jnc	.Ldone
	addq	$25,%rax
	.byte	0xf3,0xc3
.L8xchar:
	addq	$12,%rax
.Ldone:
	.byte	0xf3,0xc3
.align	64
.Lopts:
.byte	114,99,52,40,56,120,44,105,110,116,41,0  # data '114' will be disassembled as 'jb'
.byte	114,99,52,40,56,120,44,99,104,97,114,41,0
.byte	114,99,52,40,49,54,120,44,105,110,116,41,0
.byte	82,67,52,32,102,111,114,32,120,56,54,95,54,52,44,32,67,82,89,80,84,79,71,65,77,83,32,98,121,32,60,97,112,112,114,111,64,111,112,101,110,115,115,108,46,111,114,103,62,0
.align	64
.size	RC4_options,.-RC4_options
